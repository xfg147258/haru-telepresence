"""Base webcam-mode integrated system for the Haru 2.0 robot.

This module wires together face/iris extraction, head-pose tracking,
expression recognition, eye tracking with VOR compensation, robot motor
control, and realtime speech recognition into a single ROS 2 node. The
teleconference variant subclasses this module.
"""

import time
from queue import Empty, Queue
from threading import Lock, Thread

import cv2
import numpy as np
import rclpy
import torch
from rclpy.node import Node

from constants import (
    MAX_BASE_RANGE,
    MAX_HEAD_ROTATION_ANGLE,
    MAX_NECK_PITCH_RANGE,
    MAX_NECK_ROLL_RANGE,
    MOVEMENT_THRESHOLD,
    REQUIRED_CALIBRATION_FRAMES,
    ROUTINE_DURATION,
    SCALE_FACTOR_BASE,
    SCALE_FACTOR_PITCH,
    SCALE_FACTOR_ROLL,
    SPEECH_TIMEOUT,
    STABLE_THRESHOLD,
)
from expression_manager import ExpressionManager
from eye_tracker import EyeTracker
from face_eye_extractor import FaceExtractor
from robot_control import RobotController, TTSPublisher
from speech_recognizer import RealtimeStreamingSpeechRecognizer

# Frame buffer / timing.
_FRAME_QUEUE_MAX_SIZE = 5
_CAPTURE_LOOP_SLEEP_S = 0.01
_PROCESS_TIMER_PERIOD_S = 0.01
_FRAME_QUEUE_GET_TIMEOUT_S = 1.0

# Camera defaults.
_CAMERA_FRAME_WIDTH = 640
_CAMERA_FRAME_HEIGHT = 480

# Post-routine cool-down before re-entering expression recognition.
_POST_ROUTINE_COOLDOWN_S = 2.5

# Visualization rows.
_VIS_TEXT_X = 10


class HaruIntegratedSystem(Node):
    """ROS 2 node implementing the standard webcam telepresence pipeline.

    States:
        TRACKING — neutral; track head and eyes, optionally enter recognition.
        EXPRESSION_RECOGNITION — buffer landmarks; classify after enough frames.
        SPEECH_SYNC — TTS active; head/eye tracking suspended for lipsync.
    """

    def __init__(self) -> None:
        super().__init__('haru_integrated_system')

        self.running = True
        self.timer = None
        self.lock = Lock()

        # Camera.
        self.cap: cv2.VideoCapture | None = None
        self._init_camera()

        # Subsystems.
        self.face_extractor = FaceExtractor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Using device: {self.device}')

        self.robot_controller = RobotController(self)
        self.tts_publisher = TTSPublisher(self)
        self.expression_manager = ExpressionManager(self, self.device)
        self.eye_tracker = EyeTracker()

        self._init_state_variables()

        # Speech recognition.
        self.speech_recognizer: RealtimeStreamingSpeechRecognizer | None = None
        self.speech_thread: Thread | None = None
        self.speech_active = False
        self.last_speech_time = time.time()
        self.speech_timeout = SPEECH_TIMEOUT

        self.frame_queue: Queue = Queue(maxsize=_FRAME_QUEUE_MAX_SIZE)

    # ---------------------------------------------------------------------
    # Initialization helpers
    # ---------------------------------------------------------------------

    def _init_state_variables(self) -> None:
        # State machine.
        self.current_state = 'TRACKING'
        self.last_expression_time = time.time()
        self.last_head_movement = time.time()
        self.head_stable_duration = 0.0
        self.stable_threshold = STABLE_THRESHOLD

        # Head motion.
        self.prev_head_position: np.ndarray | None = None
        self.movement_threshold = MOVEMENT_THRESHOLD
        self.max_head_rotation_angle = MAX_HEAD_ROTATION_ANGLE
        self.current_head_rotation = 0.0
        self.scale_factor = SCALE_FACTOR_BASE

        # Calibration.
        self.baseline_calibrated = False
        self.baseline_horizontal_offset = 0.0
        self.baseline_vertical_offset = 0.0
        self.baseline_roll_angle = 0.0
        self.calibration_frames: list[tuple[float, float, float]] = []
        self.calibration_count = 0
        self.required_calibration_frames = REQUIRED_CALIBRATION_FRAMES

    def _init_camera(self) -> None:
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.get_logger().error('Failed to open webcam.')
                return
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, _CAMERA_FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _CAMERA_FRAME_HEIGHT)
            self.get_logger().info('Camera initialized.')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'Camera init failed: {exc}')
            self.cap = None

    # ---------------------------------------------------------------------
    # Head pose
    # ---------------------------------------------------------------------

    @staticmethod
    def calculate_head_pose(face_features) -> tuple[float, float, float]:
        """Returns (horizontal_offset, vertical_offset, roll_angle).

        Uses nose-tip / eye-line geometry to avoid full PnP solving.
        """
        if face_features is None:
            return 0.0, 0.0, 0.0

        nose_tip = face_features[1]
        left_eye = face_features[133]
        right_eye = face_features[362]
        forehead = face_features[9]
        chin = face_features[175]
        mouth_center = face_features[13]

        eye_center = (left_eye + right_eye) / 2

        horizontal_offset = nose_tip[0] - eye_center[0]
        eye_mouth_mid = (eye_center + mouth_center) / 2
        vertical_offset = nose_tip[1] - eye_mouth_mid[1]

        eye_line = right_eye - left_eye
        roll_from_eyes = np.arctan2(eye_line[1], eye_line[0])
        face_axis = chin - forehead
        roll_from_face = np.arctan2(face_axis[0], face_axis[1])
        roll_angle = 0.7 * roll_from_eyes + 0.3 * roll_from_face

        return horizontal_offset, vertical_offset, roll_angle

    def calibrate_baseline(self, face_features) -> bool:
        """Adds one calibration frame; returns True on completion."""
        if face_features is None:
            return False

        h_offset, v_offset, r_angle = self.calculate_head_pose(face_features)
        self.calibration_frames.append((h_offset, v_offset, r_angle))
        self.calibration_count = len(self.calibration_frames)

        if self.calibration_count < self.required_calibration_frames:
            return False

        # Eq. (12): the calibration reference is the per-axis median over the
        # collected frames (robust to transient outliers), not the mean.
        data = np.asarray(self.calibration_frames)
        self.baseline_horizontal_offset = float(np.median(data[:, 0]))
        self.baseline_vertical_offset = float(np.median(data[:, 1]))
        self.baseline_roll_angle = float(np.median(data[:, 2]))
        self.baseline_calibrated = True
        self.get_logger().info('Baseline calibration complete.')
        return True

    def get_calibrated_head_pose(self, face_features) -> tuple[float, float, float]:
        if face_features is None:
            return 0.0, 0.0, 0.0
        h, v, r = self.calculate_head_pose(face_features)
        if not self.baseline_calibrated:
            return h, v, r
        return (h - self.baseline_horizontal_offset,
                v - self.baseline_vertical_offset,
                r - self.baseline_roll_angle)

    # ---------------------------------------------------------------------
    # Head motion + tracking
    # ---------------------------------------------------------------------

    def detect_head_movement(self, face_features) -> float:
        nose_tip = face_features[1]
        if self.prev_head_position is None:
            self.prev_head_position = nose_tip
            return 0.0
        movement = float(np.linalg.norm(nose_tip[:2] - self.prev_head_position[:2]))
        self.prev_head_position = nose_tip
        return movement

    def track_head(self, face_features, image_shape) -> None:
        """Maps calibrated head pose + iris gaze to motor commands."""
        with self.lock:
            if not self.baseline_calibrated:
                self.get_logger().warn('Tracking called before calibration.')
                return

            h_offset, v_offset, roll_angle = self.get_calibrated_head_pose(face_features)
            height, width = image_shape[:2]

            # Eq. (13)-(15): map the calibration-relative head offsets to joint
            # angles with per-axis gains k_y / k_p / k_r. (Webcam mode uses the
            # fixed gains; the teleconference variant additionally scales them by
            # the image-sharpness factor of Eq. (16)-(17).)
            base_position = -float(np.clip(
                h_offset / width * SCALE_FACTOR_BASE * MAX_BASE_RANGE,
                -MAX_BASE_RANGE, MAX_BASE_RANGE,
            ))
            neck_pitch = float(np.clip(
                v_offset / height * SCALE_FACTOR_PITCH * MAX_NECK_PITCH_RANGE,
                -MAX_NECK_PITCH_RANGE, MAX_NECK_PITCH_RANGE,
            ))
            neck_roll = -float(np.clip(
                roll_angle * SCALE_FACTOR_ROLL,
                -MAX_NECK_ROLL_RANGE, MAX_NECK_ROLL_RANGE,
            ))

            iris_data = self.face_extractor.get_iris_data(face_features, time.time())
            left_x, left_y, right_x, right_y = self.eye_tracker.calculate_eye_position(
                face_features, iris_data, base_position, neck_pitch, neck_roll,
            )

            self.current_head_rotation = abs(base_position * 180.0 / np.pi)

            self.robot_controller.move_base(base_position)
            self.robot_controller.move_neck_pitch(neck_pitch)
            self.robot_controller.move_neck_roll(neck_roll)
            self.robot_controller.move_eyes(0, left_x, left_y)
            self.robot_controller.move_eyes(1, right_x, right_y)

    # ---------------------------------------------------------------------
    # Speech recognition
    # ---------------------------------------------------------------------

    def init_speech_recognition(self) -> None:
        try:
            def callback(text: str, language_code: str) -> None:
                self.tts_publisher.publish_tts(text, language_code)
                self.speech_active = True
                self.last_speech_time = time.time()
                self.current_state = 'SPEECH_SYNC'

            self.speech_recognizer = RealtimeStreamingSpeechRecognizer(
                callback=callback, model_type='small',
                language='en', engine='whisper',
            )
            self.speech_thread = Thread(
                target=self.speech_recognizer.start_realtime_recognition,
                daemon=True,
            )
            self.speech_thread.start()
            self.get_logger().info('Speech recognition started.')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'Speech recognition init failed: {exc}')
            self.speech_recognizer = None

    def check_speech_timeout(self) -> None:
        if (self.speech_active
                and time.time() - self.last_speech_time > self.speech_timeout):
            self.speech_active = False
            self.current_state = 'TRACKING'

    # ---------------------------------------------------------------------
    # Frame source
    # ---------------------------------------------------------------------

    def get_frame(self):
        if self.cap is None:
            return None
        ok, frame = self.cap.read()
        return frame if ok else None

    def capture_loop(self) -> None:
        """Background thread that pushes the latest frame onto the queue."""
        while self.running and rclpy.ok():
            frame = self.get_frame()
            if frame is not None:
                if not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except Empty:
                        pass
                self.frame_queue.put(frame)
            time.sleep(_CAPTURE_LOOP_SLEEP_S)

    # ---------------------------------------------------------------------
    # Visualization (split into focused helpers)
    # ---------------------------------------------------------------------

    def visualize_frame(self, frame, face_features, vis_points=None):
        self._draw_state(frame)
        self._draw_iris(frame, face_features)
        self._draw_vor(frame)
        return frame

    def _draw_state(self, frame) -> None:
        cv2.putText(frame, f'State: {self.current_state}', (_VIS_TEXT_X, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if not self.baseline_calibrated:
            cv2.putText(
                frame,
                f'Calibrating: {self.calibration_count}/'
                f'{self.required_calibration_frames}',
                (_VIS_TEXT_X, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )
        else:
            cv2.putText(frame, 'Calibrated', (_VIS_TEXT_X, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if self.speech_active:
            cv2.putText(frame, 'SPEECH SYNC ACTIVE', (_VIS_TEXT_X, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if self.speech_recognizer:
            cv2.putText(
                frame, f'Speech count: {self.speech_recognizer.sent_count}',
                (_VIS_TEXT_X, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
            )

    def _draw_iris(self, frame, face_features) -> None:
        if face_features is None or len(face_features) < 478:
            return
        iris_data = self.face_extractor.get_iris_data(face_features, time.time())
        if not iris_data:
            return

        ox, oy = iris_data['avg_iris_offset_raw']
        cv2.putText(frame, f'Iris: ({ox:.1f}, {oy:.1f})', (_VIS_TEXT_X, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
        if iris_data['micromovement_active']:
            cv2.putText(frame, 'MICROSACCADE', (_VIS_TEXT_X, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def _draw_vor(self, frame) -> None:
        status = self.eye_tracker.get_vor_status()
        is_active = status['is_active']
        velocities = status['velocities']

        color = (0, 255, 255) if is_active else (128, 128, 128)
        text = 'VOR: ACTIVE' if is_active else 'VOR: IDLE'
        cv2.putText(frame, text, (_VIS_TEXT_X, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        vel_text = (f"Vel: Y={velocities['base']:.1f} "
                    f"P={velocities['pitch']:.1f} R={velocities['roll']:.1f}")
        cv2.putText(frame, vel_text, (_VIS_TEXT_X, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)

        cv2.putText(frame, 'VOR: PG-LSTM (dual-pathway)', (_VIS_TEXT_X, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 100), 1)

    # ---------------------------------------------------------------------
    # Frame processing pipeline
    # ---------------------------------------------------------------------

    def process_frame(self) -> None:
        """Single tick of the main processing loop."""
        try:
            frame = self.frame_queue.get(timeout=_FRAME_QUEUE_GET_TIMEOUT_S)
        except Empty:
            return

        face_features = self.face_extractor.extract(frame)
        rep_keypoints, vis_points = (None, None)
        if face_features is not None:
            rep_keypoints, vis_points = self.expression_manager.extract_landmarks(frame)

        self.expression_manager.check_routine_timeout()
        self.check_speech_timeout()

        # Calibration phase: gather frames; show a progress overlay only.
        if not self.baseline_calibrated and face_features is not None:
            if self.calibrate_baseline(face_features):
                self.robot_controller.reset_motors()
                time.sleep(0.5)
                self.init_speech_recognition()
            self._show(self.visualize_frame(frame, face_features, vis_points))
            return

        # Routine completion transition.
        if self.expression_manager.is_routine_finished(ROUTINE_DURATION):
            self.current_state = 'SPEECH_SYNC' if self.speech_active else 'TRACKING'

        if face_features is not None:
            self._run_state_machine(face_features, frame, rep_keypoints)

        self._show(self.visualize_frame(frame, face_features, vis_points))

    def _run_state_machine(self, face_features, frame, rep_keypoints) -> None:
        """Dispatches state transitions and per-state actions."""
        em = self.expression_manager

        # Pause head tracking while a routine is queued or running.
        if not em.executing_routine and not em.has_pending_routine:
            self.track_head(face_features, frame.shape)

        head_movement = self.detect_head_movement(face_features)
        now = time.time()

        if em.executing_routine or em.has_pending_routine:
            return

        if self.current_state == 'TRACKING':
            self._handle_tracking(head_movement, now)
        elif self.current_state == 'EXPRESSION_RECOGNITION':
            self._handle_expression(head_movement, now, rep_keypoints)
        elif self.current_state == 'SPEECH_SYNC' and not self.speech_active:
            self.current_state = 'TRACKING'

    def _handle_tracking(self, head_movement: float, now: float) -> None:
        em = self.expression_manager
        if head_movement >= self.movement_threshold:
            self.head_stable_duration = 0.0
            self.last_head_movement = now
            return

        self.head_stable_duration = now - self.last_head_movement

        cooldown_passed = (
            em.routine_start_time is None
            or now - em.routine_start_time > _POST_ROUTINE_COOLDOWN_S
        )
        if (self.head_stable_duration > self.stable_threshold
                and cooldown_passed
                and self.current_head_rotation < self.max_head_rotation_angle
                and not self.speech_active):
            self.current_state = 'EXPRESSION_RECOGNITION'

    def _handle_expression(self, head_movement: float, now: float,
                           rep_keypoints) -> None:
        em = self.expression_manager
        if (head_movement > self.movement_threshold
                or self.current_head_rotation >= self.max_head_rotation_angle
                or self.speech_active):
            self.current_state = 'SPEECH_SYNC' if self.speech_active else 'TRACKING'
            self.head_stable_duration = 0.0
            self.last_head_movement = now
            return

        if rep_keypoints is None or self.speech_active:
            return

        em.add_frame_to_sequence(rep_keypoints)
        if len(em.frame_sequence) < 4:
            return

        routine_id = em.recognize_expression()
        if routine_id is not None:
            em.execute_routine(routine_id)
        else:
            self.current_state = 'TRACKING'
            em.reset_sequence()

    @staticmethod
    def _show(frame) -> None:
        cv2.imshow('Haru 2.0 System', frame)
        cv2.waitKey(1)

    # ---------------------------------------------------------------------
    # Timer + lifecycle
    # ---------------------------------------------------------------------

    def start_processing(self) -> None:
        self.timer = self.create_timer(_PROCESS_TIMER_PERIOD_S, self._timer_callback)

    def _timer_callback(self) -> None:
        if self.running:
            self.process_frame()

    def run(self) -> None:
        self.get_logger().info('Starting integrated system...')
        self.robot_controller.reset_motors()
        self.get_logger().info('Motors reset; waiting for baseline calibration.')

        self.start_processing()

        capture_thread = Thread(target=self.capture_loop, daemon=True)
        capture_thread.start()

        try:
            rclpy.spin(self)
        except KeyboardInterrupt:
            self.get_logger().info('Interrupted by user.')
        finally:
            self.running = False
            if self.speech_recognizer:
                self.speech_recognizer.stop()
            capture_thread.join(timeout=2.0)
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
