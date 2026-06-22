"""Teleconference (screen-capture) variant of the integrated Haru system."""

import time
from threading import Thread

import cv2
import numpy as np
import rclpy

from base_system import HaruIntegratedSystem
from constants import (
    TELECONFERENCE_CALIBRATION_FRAMES,
    TELECONFERENCE_TARGET_SIZE,
)
from tracking import EnhancedCalibration, ImprovedTeleconferenceTracking

_COUNTDOWN_SECONDS = 5
_CAPTURE_THREAD_JOIN_TIMEOUT_S = 2.0


class HaruTeleconferenceSystem(HaruIntegratedSystem):
    """Streams a remote participant's video, tracks their head, and mirrors it.

    Differences vs. HaruIntegratedSystem:
        * Frame source is screen capture (mss) instead of a local webcam.
        * Head pose is estimated by ImprovedTeleconferenceTracking, which is
          more tolerant of low-resolution/laggy remote video.
        * Speech recognition runs on system-monitor audio (other participant).
    """

    def __init__(self) -> None:
        super().__init__()

        self._screen_capture, self._mouse = self._import_capture_modules()

        self.capture_mode = 'screen'
        self.monitor: dict | None = None

        self.improved_tracker = ImprovedTeleconferenceTracking()
        self.enhanced_calibrator = EnhancedCalibration(
            required_frames=TELECONFERENCE_CALIBRATION_FRAMES
        )

        self.scale_factor = self.improved_tracker.scale_factor_base
        self.required_calibration_frames = TELECONFERENCE_CALIBRATION_FRAMES

        self._close_camera()

    # ---------------------------------------------------------------------
    # Setup helpers
    # ---------------------------------------------------------------------

    def _import_capture_modules(self):
        """Imports `mss` and `pyautogui`; logs and returns (None, None) on failure."""
        try:
            import mss
            import pyautogui
            self.get_logger().info('Screen-capture modules loaded.')
            return mss, pyautogui
        except ImportError as exc:
            self.get_logger().error(f'Failed to import screen-capture modules: {exc}')
            return None, None

    def _close_camera(self) -> None:
        """Releases the local webcam if the base class opened it."""
        try:
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
                self.cap = None
                self.get_logger().info('Webcam released; switching to screen capture.')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'Error releasing webcam: {exc}')
            self.cap = None

    def _setup_screen_capture(self) -> bool:
        """Interactively prompts the user to select a screen region with the mouse."""
        if self._mouse is None:
            self.get_logger().error('pyautogui unavailable; cannot set up capture.')
            return False

        try:
            print('\n' + '=' * 60)
            print('  Screen capture setup')
            print('=' * 60)
            print('1. Open your video conference app.')
            print('2. Make the remote participant visible.')
            print('3. Move the mouse to the corners as prompted.\n')

            self._countdown(_COUNTDOWN_SECONDS,
                            'Move the mouse to the TOP-LEFT corner of the video window')
            x1, y1 = self._mouse.position()
            print(f'  Top-left: ({x1}, {y1})')

            self._countdown(_COUNTDOWN_SECONDS,
                            'Move the mouse to the BOTTOM-RIGHT corner of the video window')
            x2, y2 = self._mouse.position()
            print(f'  Bottom-right: ({x2}, {y2})')

            self.monitor = {
                'top': min(y1, y2),
                'left': min(x1, x2),
                'width': abs(x2 - x1),
                'height': abs(y2 - y1),
            }
            print(f"\nCapture region: ({self.monitor['left']}, {self.monitor['top']}), "
                  f"{self.monitor['width']}x{self.monitor['height']}")
            return True
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'Screen capture setup failed: {exc}')
            return False

    @staticmethod
    def _countdown(seconds: int, message: str) -> None:
        for i in range(seconds, 0, -1):
            print(f'{message} — starting in {i}s...')
            time.sleep(1)

    # ---------------------------------------------------------------------
    # Frame source
    # ---------------------------------------------------------------------

    def get_frame(self) -> np.ndarray | None:
        """Grabs a frame from the configured screen region and crops to a face."""
        if (self.capture_mode != 'screen'
                or self.monitor is None
                or self._screen_capture is None):
            return None
        try:
            with self._screen_capture.mss() as sct:
                shot = sct.grab(self.monitor)
                frame = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)

            face_crop, detected = self.improved_tracker.detect_and_crop_face(frame)
            if detected:
                return face_crop
            return cv2.resize(frame, TELECONFERENCE_TARGET_SIZE)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'Screen capture failed: {exc}')
            return None

    # ---------------------------------------------------------------------
    # Head pose & tracking overrides
    # ---------------------------------------------------------------------

    def calculate_head_pose(self, face_features) -> tuple[float, float, float]:
        if face_features is None:
            return 0.0, 0.0, 0.0

        h_offset, v_offset, roll = (
            self.improved_tracker.improved_head_pose_calculation(face_features))

        if not self.baseline_calibrated:
            if self.enhanced_calibrator.add_frame(h_offset, v_offset, roll):
                self.baseline_calibrated = True
                self.get_logger().info('Enhanced calibration complete.')
            return 0.0, 0.0, 0.0

        return self.enhanced_calibrator.apply_calibration(h_offset, v_offset, roll)

    def get_calibrated_head_pose(self, face_features):
        return self.calculate_head_pose(face_features)

    def calibrate_baseline(self, face_features) -> bool:
        if face_features is None:
            return False
        h_offset, v_offset, roll = (
            self.improved_tracker.improved_head_pose_calculation(face_features))
        if self.enhanced_calibrator.add_frame(h_offset, v_offset, roll):
            self.baseline_calibrated = True
            return True
        return False

    def track_head(self, face_features, image_shape) -> None:
        with self.lock:
            if not self.baseline_calibrated:
                return

            h_offset, v_offset, roll = self.calculate_head_pose(face_features)

            frame = self.get_frame()
            scale_factors = (
                self.improved_tracker.calculate_adaptive_scale_factors(frame)
                if frame is not None else
                {
                    'base': self.improved_tracker.scale_factor_base,
                    'pitch': self.improved_tracker.scale_factor_pitch,
                    'roll': self.improved_tracker.scale_factor_roll,
                }
            )

            base_pos, neck_pitch, neck_roll = (
                self.improved_tracker.map_to_robot_commands(
                    h_offset, v_offset, roll, image_shape, scale_factors))

            iris_data = self.face_extractor.get_iris_data(face_features, time.time())
            left_x, left_y, right_x, right_y = (
                self.eye_tracker.calculate_eye_position(
                    face_features, iris_data, base_pos, neck_pitch, neck_roll))

            self.current_head_rotation = abs(base_pos * 180.0 / np.pi)

            self.robot_controller.move_base(base_pos)
            self.robot_controller.move_neck_pitch(neck_pitch)
            self.robot_controller.move_neck_roll(neck_roll)
            self.robot_controller.move_eyes(0, left_x, left_y)
            self.robot_controller.move_eyes(1, right_x, right_y)

    def init_speech_recognition(self) -> None:
        """Speech recognition is provided by the remote-audio recognizer."""
        self.get_logger().info('Speech recognition disabled at base level '
                               '(handled by remote audio recognizer).')
        self.speech_recognizer = None

    # ---------------------------------------------------------------------
    # Visualization (split into focused helpers)
    # ---------------------------------------------------------------------

    def visualize_frame(self, frame, face_features, vis_points=None):
        self._draw_state(frame)
        self._draw_iris_overlay(frame, face_features)
        self._draw_vor_overlay(frame)
        self._draw_teleconf_status(frame)
        return frame

    def _draw_state(self, frame) -> None:
        cv2.putText(frame, f'State: {self.current_state}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if not self.baseline_calibrated:
            cv2.putText(
                frame,
                f'Calibrating: {self.calibration_count}/'
                f'{self.required_calibration_frames}',
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )
        else:
            cv2.putText(frame, 'Calibrated', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if self.speech_active:
            cv2.putText(frame, 'SPEECH SYNC ACTIVE', (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if self.speech_recognizer:
            cv2.putText(
                frame, f'Speech count: {self.speech_recognizer.sent_count}',
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
            )

    def _draw_iris_overlay(self, frame, face_features) -> None:
        if face_features is None or len(face_features) < 478:
            return
        iris_data = self.face_extractor.get_iris_data(face_features, time.time())
        if not iris_data:
            return

        ox, oy = iris_data['avg_iris_offset_raw']
        cv2.putText(frame, f'Iris: ({ox:.1f}, {oy:.1f})', (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)

        mx, my = iris_data['micromovement_offset']
        cv2.putText(frame, f'Micro: ({mx:.2f}, {my:.2f})', (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 100), 2)

        if iris_data['micromovement_active']:
            cv2.putText(frame, 'MICROSACCADE!', (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def _draw_vor_overlay(self, frame) -> None:
        status = self.eye_tracker.get_vor_status()
        is_active = status['is_active']
        velocities = status['velocities']

        color = (0, 255, 255) if is_active else (128, 128, 128)
        text = 'VOR: ACTIVE' if is_active else 'VOR: IDLE'
        cv2.putText(frame, text, (10, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        vel_text = (f"Vel: Y={velocities['base']:.1f} "
                    f"P={velocities['pitch']:.1f} R={velocities['roll']:.1f}")
        cv2.putText(frame, vel_text, (10, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)

        cv2.putText(frame, 'VOR: PG-LSTM (dual-pathway)', (10, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 100), 1)

    def _draw_teleconf_status(self, frame) -> None:
        cv2.putText(frame, 'Mode: Teleconference', (10, 330),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        face_text = ('Face: Detected & Cropped'
                     if self.improved_tracker.face_roi is not None
                     else 'Face: Searching...')
        face_color = ((0, 255, 0)
                      if self.improved_tracker.face_roi is not None
                      else (0, 165, 255))
        cv2.putText(frame, face_text, (10, 360),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, face_color, 2)

        audio_active = (
            self.speech_recognizer is not None
            and hasattr(self.speech_recognizer, 'audio_manager')
            and self.speech_recognizer.audio_manager.monitor_source is not None
        )
        audio_text = 'Remote Audio: Active' if audio_active else 'Remote Audio: N/A'
        audio_color = (0, 255, 0) if audio_active else (128, 128, 128)
        cv2.putText(frame, audio_text, (10, 390),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, audio_color, 2)

    # ---------------------------------------------------------------------
    # Main loop
    # ---------------------------------------------------------------------

    def run(self) -> None:
        self.get_logger().info('Starting teleconference system...')
        self.robot_controller.reset_motors()
        self.get_logger().info('Motors reset; starting screen-capture setup.')

        if not self._setup_screen_capture():
            self.get_logger().error('Screen capture setup failed; exiting.')
            return

        self.get_logger().info('Screen capture ready; awaiting calibration.')
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
            capture_thread.join(timeout=_CAPTURE_THREAD_JOIN_TIMEOUT_S)
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
