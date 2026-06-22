"""Training-data collector for the Haru 2.0 facial-expression model.

Two capture modes are supported:

* camera         — use the local webcam.
* teleconference — capture a region of the screen (Zoom, Meet, etc.).

For each routine ID in ROUTINE_IDS the script: triggers the routine on
the robot, waits ROUTINE_EXEC_DELAY_S, samples 226-D face landmarks for
SAMPLE_DURATION_S, optionally collects a few pose-variation samples, and
finally augments the captured set. Everything is saved to an .npz file
under data/.

Examples:
    python collect_data.py
    python collect_data.py teleconference
"""

import json
import os
import queue
import random
import sys
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
import rclpy
from cv_bridge import CvBridge
from haru2_core_msgs.msg import EyesPose, LCDCommand, LedsArray, MotorCommand
from haru2_core_msgs.srv import Routine
from rclpy.node import Node
from std_msgs.msg import Empty, String

from representative_keypoints import get_visualization_points, representative_keypoints

try:
    import mss
    import pyautogui
    _SCREEN_CAPTURE_AVAILABLE = True
except ImportError:
    _SCREEN_CAPTURE_AVAILABLE = False

# =============================================================================
# Constants
# =============================================================================

ROUTINE_IDS = [
    2092, 2083, 2068, 2057, 2006, 2037,
    2036, 2087, 2010, 2071, 2073, 2023,
    2081, 2021, 2067, 2059, 2017,
]

# Capture / sampling.
_FRAME_WIDTH = 640
_FRAME_HEIGHT = 480
_LANDMARK_RESIZE = (480, 320)
_SAMPLE_DURATION_S = 3.0
_SAMPLE_INTERVAL_S = 0.1
_POSE_VARIATION_DURATION_S = 3.0
_ROUTINE_EXEC_DELAY_S = 4.0
_MOTOR_RESET_DELAY_S = 2.0

# Augmentation.
_AUG_RATIO = 0.6
_AUG_VARIATIONS = 4
_POSE_VARIATIONS_PER_SAMPLE = 2

_SCREEN_CAPTURE_PROMPT_COUNTDOWN_S = 5

_POSITION_PROMPTS = (
    'Hold the expression — move slightly closer to the camera.',
    'Hold the expression — move slightly away from the camera.',
    'Hold the expression — tilt your head slightly to the left.',
    'Hold the expression — tilt your head slightly to the right.',
    'Hold the expression — raise your head slightly.',
    'Hold the expression — lower your head slightly.',
)

# Output paths.
_OUTPUT_DIR = 'data'
_CAMERA_OUTPUT = 'haru2_camera_training_data.npz'
_TELECONFERENCE_OUTPUT = 'haru2_teleconference_training_data.npz'


# =============================================================================
# Augmentation
# =============================================================================

def _augment_keypoints(keypoints: np.ndarray, variations: int) -> list[np.ndarray]:
    """Generates `variations` random augmentations of a 226-D keypoint vector."""
    out = []
    for _ in range(variations):
        aug = keypoints.copy()
        kind = random.choice(
            ('noise', 'scale', 'shift', 'combined', 'distance_sim', 'pose_sim'))

        if kind in ('noise', 'combined'):
            aug = aug + np.random.normal(
                0, random.uniform(0.002, 0.012), aug.shape)

        if kind in ('scale', 'combined', 'distance_sim'):
            pts = aug.reshape(-1, 2)
            cx, cy = pts.mean(axis=0)
            pts -= [cx, cy]
            pts *= 1.0 + random.uniform(-0.15, 0.15)
            pts += [cx, cy]
            aug = pts.flatten()

        if kind in ('shift', 'combined', 'pose_sim'):
            pts = aug.reshape(-1, 2)
            pts[:, 0] += random.uniform(-0.03, 0.03)
            pts[:, 1] += random.uniform(-0.03, 0.03)
            aug = pts.flatten()

        if kind == 'pose_sim':
            pts = aug.reshape(-1, 2)
            angle = random.uniform(-15, 15) * np.pi / 180
            cos, sin = np.cos(angle), np.sin(angle)
            rot = np.array([[cos, -sin], [sin, cos]])
            center = pts.mean(axis=0)
            aug = ((pts - center) @ rot.T + center).flatten()

        out.append(np.clip(aug, 0.0, 1.0))
    return out


# =============================================================================
# Data collector
# =============================================================================

class EnhancedDataCollector(Node):
    """ROS 2 node that drives expression-routine sampling and persistence."""

    def __init__(self, capture_mode: str = 'camera') -> None:
        super().__init__('haru2_enhanced_data_collector')
        self.bridge = CvBridge()
        self.capture_mode = capture_mode
        self.get_logger().info(f'Capture mode: {self.capture_mode}')

        self._init_publishers()
        self._init_face_mesh()

        self.cap: cv2.VideoCapture | None = None
        self.monitor: dict | None = None
        self._init_capture()

        # Sample buffers.
        self.landmark_list: list[np.ndarray] = []
        self.routine_id_list: list[int] = []
        self.available_routines = ROUTINE_IDS.copy()
        random.shuffle(self.available_routines)
        self.routine_samples_count = {rid: 0 for rid in ROUTINE_IDS}

        # Service client.
        self.routine_client = self.create_client(Routine, '/haru2/execute_routine')

        # Non-blocking input.
        self.input_queue: queue.Queue = queue.Queue()
        self.input_thread: threading.Thread | None = None

        # OpenCV window.
        self.window_name = (f'Haru2.0 Data Collection — '
                            f'{self.capture_mode.capitalize()} mode')
        self.window_created = False

        self.current_state = 'idle'

    # ---------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------

    def _init_publishers(self) -> None:
        self.routine_pub = self.create_publisher(String, '/haru2/set_routine_file', 10)
        self.pub_base = self.create_publisher(MotorCommand, '/haru2/cmd_base_pos', 10)
        self.pub_neck_pitch = self.create_publisher(
            MotorCommand, '/haru2/cmd_neck_pitch_pos', 10)
        self.pub_neck_roll = self.create_publisher(
            MotorCommand, '/haru2/cmd_neck_roll_pos', 10)
        self.pub_eyes_pose = self.create_publisher(EyesPose, '/haru2/cmd_eyes_pose', 10)
        self.pub_mouth_leds = self.create_publisher(
            LedsArray, '/haru2/cmd_mouth_leds', 10)
        self.pub_lcd = self.create_publisher(LCDCommand, '/haru2/cmd_lcd', 10)
        self.pub_all_home = self.create_publisher(Empty, '/haru2/cmd_move_home', 10)

    def _init_face_mesh(self) -> None:
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True)

    def _init_capture(self) -> bool:
        if self.capture_mode == 'camera':
            return self._init_camera()
        if self.capture_mode == 'teleconference':
            if not _SCREEN_CAPTURE_AVAILABLE:
                self.get_logger().error(
                    'Screen capture unavailable; install mss and pyautogui.')
                return False
            return self._init_screen_capture()
        return False

    def _init_camera(self) -> bool:
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.get_logger().error('Failed to open webcam.')
                return False
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, _FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _FRAME_HEIGHT)
            self.get_logger().info('Camera initialized.')
            return True
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'Camera init failed: {exc}')
            return False

    def _init_screen_capture(self) -> bool:
        try:
            self.get_logger().info('Setting up screen-capture region...')

            def countdown(seconds: int, message: str) -> None:
                for i in range(seconds, 0, -1):
                    print(f'{message} — starting in {i}s...')
                    time.sleep(1)

            countdown(_SCREEN_CAPTURE_PROMPT_COUNTDOWN_S,
                      'Move the cursor to the TOP-LEFT of the call window')
            x1, y1 = pyautogui.position()
            print(f'Top-left recorded: ({x1}, {y1})')

            countdown(_SCREEN_CAPTURE_PROMPT_COUNTDOWN_S,
                      'Move the cursor to the BOTTOM-RIGHT of the call window')
            x2, y2 = pyautogui.position()
            print(f'Bottom-right recorded: ({x2}, {y2})')

            self.monitor = {
                'top': min(y1, y2),
                'left': min(x1, x2),
                'width': abs(x2 - x1),
                'height': abs(y2 - y1),
            }
            self.get_logger().info(f'Screen-capture region: {self.monitor}')
            return True
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'Screen-capture init failed: {exc}')
            return False

    # ---------------------------------------------------------------------
    # Frame source
    # ---------------------------------------------------------------------

    def _get_frame(self):
        if self.capture_mode == 'camera':
            if self.cap is None:
                return None
            ok, frame = self.cap.read()
            return frame if ok else None

        if self.monitor is None:
            return None
        try:
            with mss.mss() as sct:
                shot = sct.grab(self.monitor)
                return cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warning(f'Screen grab failed: {exc}')
            return None

    # ---------------------------------------------------------------------
    # Landmarks + visualization
    # ---------------------------------------------------------------------

    def _extract_landmarks(self, frame):
        small = cv2.resize(frame, _LANDMARK_RESIZE, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None, None

        lmks = []
        for lm in results.multi_face_landmarks[0].landmark:
            lmks.extend([lm.x, lm.y])
        landmarks = np.array(lmks)
        if landmarks.shape[0] < 936:
            return None, None
        return representative_keypoints(landmarks), get_visualization_points(landmarks)

    def _draw_frame(self, frame, rep_keypoints, vis_points,
                    message: str, remaining_time: float | None = None):
        if frame is None:
            return None
        out = frame.copy()
        h, w, _ = out.shape

        if vis_points is not None:
            for x, y in vis_points:
                cv2.circle(out, (int(x * w), int(y * h)), 2, (0, 255, 0), -1)

        cv2.putText(out, message, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(out, f'Mode: {self.capture_mode.capitalize()}',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        if remaining_time is not None:
            cv2.putText(out, f'Countdown: {remaining_time:.1f}s', (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(out, "Press 'q' to quit", (10, h - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(out, 'Use the terminal for prompts', (10, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        face_text = ('Face landmarks detected' if rep_keypoints is not None
                     else 'No face detected')
        face_color = ((0, 255, 0) if rep_keypoints is not None
                      else (0, 0, 255))
        cv2.putText(out, face_text, (10, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_color, 1)
        return out

    # ---------------------------------------------------------------------
    # OpenCV window + non-blocking input
    # ---------------------------------------------------------------------

    def _create_window(self) -> None:
        if not self.window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            self.window_created = True

    def _destroy_window(self) -> None:
        if self.window_created:
            cv2.destroyWindow(self.window_name)
            self.window_created = False

    def _start_input_thread(self, prompt: str) -> None:
        if self.input_thread and self.input_thread.is_alive():
            return

        def worker() -> None:
            try:
                print(f'\n{prompt}')
                self.input_queue.put(input().strip().lower())
            except Exception:  # noqa: BLE001
                self.input_queue.put('')

        self.input_thread = threading.Thread(target=worker, daemon=True)
        self.input_thread.start()

    def _get_user_input(self) -> str | None:
        try:
            return self.input_queue.get_nowait()
        except queue.Empty:
            return None

    # ---------------------------------------------------------------------
    # Robot control
    # ---------------------------------------------------------------------

    def _call_routine_service(self, routine_id: int) -> int | None:
        while not self.routine_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for routine service...')

        request = Routine.Request()
        request.routine = routine_id
        future = self.routine_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        try:
            future.result()
            self.get_logger().info(f'Routine {routine_id} executed.')
            return routine_id
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'Routine service call failed: {exc}')
            return None

    def _publish_routine_id(self, routine_id: int) -> None:
        msg = String()
        msg.data = json.dumps({'id': routine_id})
        self.routine_pub.publish(msg)

    def _generate_and_publish_routine(self) -> int | None:
        if not self.available_routines:
            self.available_routines = ROUTINE_IDS.copy()
            random.shuffle(self.available_routines)
            self.get_logger().info('Routine list exhausted, refilled.')

        while self.available_routines:
            rid = self.available_routines.pop()
            result = self._call_routine_service(rid)
            if result is not None:
                self._publish_routine_id(result)
                return result
            self.get_logger().warning(
                f'Routine {rid} call failed; trying another.')
        self.get_logger().error('All routine calls failed; ending.')
        return None

    def _reset_motors(self) -> None:
        self.pub_all_home.publish(Empty())
        self.get_logger().info('Resetting motors to home...')
        time.sleep(_MOTOR_RESET_DELAY_S)

    # ---------------------------------------------------------------------
    # Sample acquisition
    # ---------------------------------------------------------------------

    def _wait_for_yes(self, prompt: str, on_screen_msg: str) -> bool:
        """Blocks until the user types 'y' (returns True) or 'q' (returns False)."""
        self._start_input_thread(prompt)
        while rclpy.ok():
            frame = self._get_frame()
            if frame is not None:
                rep_keypoints, vis_points = self._extract_landmarks(frame)
                cv2.imshow(self.window_name,
                           self._draw_frame(frame, rep_keypoints, vis_points,
                                            on_screen_msg))
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                return False
            user_input = self._get_user_input()
            if user_input is None:
                continue
            if user_input == 'y':
                return True
            self._start_input_thread(
                f"Please type 'y' to start (got: {user_input}):")

    def _capture_for_duration(self, routine_id: int,
                              duration: float, message: str):
        """Captures landmarks for `duration` seconds; returns (last_kpts, last_frame)."""
        last_keypoints = None
        last_frame = None
        start = time.time()

        while (time.time() - start) < duration and rclpy.ok():
            frame = self._get_frame()
            if frame is not None:
                rep_keypoints, vis_points = self._extract_landmarks(frame)
                if rep_keypoints is not None:
                    last_keypoints = rep_keypoints
                    last_frame = self._draw_frame(
                        frame, rep_keypoints, vis_points,
                        f'Captured — Routine ID: {routine_id}')

                cv2.imshow(self.window_name,
                           self._draw_frame(frame, rep_keypoints, vis_points,
                                            message,
                                            duration - (time.time() - start)))
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                return None, None
            time.sleep(_SAMPLE_INTERVAL_S)

        return last_keypoints, last_frame

    def _confirm_sample(self, captured_frame, prompt: str) -> bool:
        """Asks the user to keep or discard the just-captured sample."""
        self._start_input_thread(prompt)
        while rclpy.ok():
            if captured_frame is not None:
                shown = captured_frame.copy()
                cv2.putText(shown, "Awaiting confirm — terminal: 'y' to keep",
                            (10, shown.shape[0] - 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow(self.window_name, shown)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                return False
            user_input = self._get_user_input()
            if user_input is not None:
                return user_input == 'y'

    def _collect_pose_variation(self, routine_id: int) -> np.ndarray | None:
        prompt_text = random.choice(_POSITION_PROMPTS)
        self.current_state = 'pose_variation'

        if not self._wait_for_yes(
                f'>>> {prompt_text} <<<\nPress y when ready:',
                f'Pose variation: {prompt_text}'):
            return None

        self.current_state = 'collecting'
        keypoints, captured_frame = self._capture_for_duration(
            routine_id, _POSE_VARIATION_DURATION_S, 'Capturing pose variation...')
        if keypoints is None:
            return None

        self.current_state = 'waiting_confirmation'
        if self._confirm_sample(
                captured_frame,
                "Keep this pose variation? Type 'y' to keep, anything else to drop:"):
            self.landmark_list.append(keypoints)
            self.routine_id_list.append(routine_id)
            self.get_logger().info(
                f'Added pose variation (Routine {routine_id}).')
            return keypoints
        self.get_logger().info('Pose variation discarded.')
        return None

    # ---------------------------------------------------------------------
    # Main collection loop
    # ---------------------------------------------------------------------

    def collect_data(self, samples_per_routine: int = 1,
                     max_samples: int = 200) -> None:
        self.get_logger().info(
            f'Starting collection: {samples_per_routine} samples per routine, '
            f'mode={self.capture_mode}.')
        self._create_window()
        self._reset_motors()

        total_samples = 0
        original_samples: list[tuple[np.ndarray, int]] = []

        try:
            while rclpy.ok() and total_samples < max_samples:
                routine_id = self._select_next_routine_id(samples_per_routine)
                if routine_id is None:
                    break
                if (self.routine_samples_count.get(routine_id, 0)
                        >= samples_per_routine):
                    self.get_logger().info(
                        f'Routine {routine_id} already complete; skipping.')
                    continue

                time.sleep(_ROUTINE_EXEC_DELAY_S)
                self.get_logger().info(
                    f'Capturing for routine {routine_id} '
                    f'({self.routine_samples_count.get(routine_id, 0)}/'
                    f'{samples_per_routine} done).')

                self.current_state = 'collecting'
                keypoints, captured_frame = self._capture_for_duration(
                    routine_id, _SAMPLE_DURATION_S,
                    f'Capturing... Routine ID: {routine_id}')
                if keypoints is None:
                    self.get_logger().warning(
                        'No valid landmarks captured; retrying.')
                    continue

                self.current_state = 'waiting_confirmation'
                keep = self._confirm_sample(
                    captured_frame,
                    f"Keep sample (routine {routine_id})? "
                    "Type 'y' to keep, anything else to retake:")
                if not keep:
                    self.get_logger().info('Sample discarded; will retake.')
                    continue

                self.landmark_list.append(keypoints)
                self.routine_id_list.append(routine_id)
                original_samples.append((keypoints, routine_id))
                total_samples += 1
                self.routine_samples_count[routine_id] = (
                    self.routine_samples_count.get(routine_id, 0) + 1)
                self.get_logger().info(
                    f'Total samples: {total_samples}; routine '
                    f'{routine_id}: {self.routine_samples_count[routine_id]}/'
                    f'{samples_per_routine}.')

                total_samples += self._maybe_collect_pose_variations(
                    routine_id, captured_frame, original_samples)

                if all(self.routine_samples_count.get(rid, 0) >= samples_per_routine
                       for rid in ROUTINE_IDS):
                    self.get_logger().info('All routines complete.')
                    break

            total_samples += self._apply_data_augmentation(original_samples)
            self._save_dataset()

        except KeyboardInterrupt:
            self.get_logger().info('Interrupted by user.')
        finally:
            self._destroy_window()
            if self.cap is not None:
                self.cap.release()

    def _select_next_routine_id(self, samples_per_routine: int) -> int | None:
        """Chooses the routine with the fewest samples; falls back to a random pick."""
        if not self.available_routines:
            return self._generate_and_publish_routine()

        sorted_routines = sorted(
            ((rid, self.routine_samples_count.get(rid, 0))
             for rid in self.available_routines),
            key=lambda x: x[1])

        if sorted_routines[0][1] >= samples_per_routine:
            return self._generate_and_publish_routine()

        min_count = sorted_routines[0][1]
        candidates = [rid for rid, count in sorted_routines if count == min_count]
        rid = random.choice(candidates)
        self.available_routines.remove(rid)
        if self._call_routine_service(rid) is None:
            return None
        self._publish_routine_id(rid)
        return rid

    def _maybe_collect_pose_variations(self, routine_id: int, captured_frame,
                                       original_samples: list) -> int:
        self._start_input_thread('Collect pose variations? (y/n):')
        added = 0
        while rclpy.ok():
            if captured_frame is not None:
                shown = captured_frame.copy()
                cv2.putText(shown, "Pose variations? terminal: 'y' or 'n'",
                            (10, shown.shape[0] - 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.imshow(self.window_name, shown)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
            user_input = self._get_user_input()
            if user_input is None:
                continue
            if user_input == 'y':
                for i in range(_POSE_VARIATIONS_PER_SAMPLE):
                    print(f'\nPose variation {i + 1}/'
                          f'{_POSE_VARIATIONS_PER_SAMPLE}')
                    sample = self._collect_pose_variation(routine_id)
                    if sample is not None:
                        original_samples.append((sample, routine_id))
                        added += 1
                    if not rclpy.ok():
                        break
            break
        return added

    def _apply_data_augmentation(self, original_samples: list) -> int:
        if not original_samples:
            return 0
        self.get_logger().info('Applying data augmentation...')
        n_to_augment = int(len(original_samples) * _AUG_RATIO)
        added = 0
        for keypoints, routine_id in random.sample(original_samples, n_to_augment):
            for aug in _augment_keypoints(keypoints, _AUG_VARIATIONS):
                self.landmark_list.append(aug)
                self.routine_id_list.append(routine_id)
                added += 1
        self.get_logger().info(f'Augmentation added {added} samples.')
        return added

    def _save_dataset(self) -> None:
        if not self.landmark_list:
            self.get_logger().warning('No samples captured; nothing to save.')
            return

        landmarks = np.array(self.landmark_list)
        routines = np.array(self.routine_id_list)
        os.makedirs(_OUTPUT_DIR, exist_ok=True)

        filename = (_TELECONFERENCE_OUTPUT
                    if self.capture_mode == 'teleconference'
                    else _CAMERA_OUTPUT)
        path = os.path.join(_OUTPUT_DIR, filename)
        np.savez(path, landmarks=landmarks, routines=routines)
        self.get_logger().info(f'Saved dataset: {path}')

        unique, counts = np.unique(routines, return_counts=True)
        self.get_logger().info(f'Per-routine sample counts ({self.capture_mode}):')
        for rid, count in zip(unique, counts):
            self.get_logger().info(f'  routine {rid}: {count} samples')


# =============================================================================
# Entry point
# =============================================================================

def main(args=None) -> None:
    capture_mode = 'camera'
    if len(sys.argv) > 1:
        if sys.argv[1] == 'teleconference':
            if not _SCREEN_CAPTURE_AVAILABLE:
                print('Screen capture unavailable. Install: pip install mss pyautogui')
                return
            capture_mode = 'teleconference'
        elif sys.argv[1] == 'camera':
            capture_mode = 'camera'
        else:
            print('Usage: python collect_data.py [camera|teleconference]')
            return

    print(f'Starting collector in {capture_mode} mode.')
    rclpy.init(args=args)
    collector = EnhancedDataCollector(capture_mode=capture_mode)
    try:
        time.sleep(2)
        collector.collect_data(samples_per_routine=2, max_samples=200)
    except KeyboardInterrupt:
        collector.get_logger().info('Interrupted by user.')
    finally:
        collector.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
