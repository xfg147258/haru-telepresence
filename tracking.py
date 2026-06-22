"""Robust head pose tracking and calibration for teleconference video."""

from collections import deque

import cv2
import mediapipe as mp
import numpy as np

from constants import (
    DEADZONE_HORIZONTAL,
    DEADZONE_ROLL,
    DEADZONE_VERTICAL,
    MAX_BASE_RANGE,
    MAX_NECK_PITCH_RANGE,
    MAX_NECK_ROLL_RANGE,
    REQUIRED_CALIBRATION_FRAMES,
    SCALE_FACTOR_BASE,
    SCALE_FACTOR_PITCH,
    SCALE_FACTOR_ROLL,
    SMOOTHING_WINDOW,
    TELECONFERENCE_ROI_PADDING,
    TELECONFERENCE_TARGET_SIZE,
)

# Landmark indices used for head-pose estimation.
_NOSE_TIP = 1
_LEFT_EYE = 133
_RIGHT_EYE = 362
_FOREHEAD = 9
_CHIN = 175
_MOUTH_CENTER = 13

# Roll fusion weights.
_ROLL_WEIGHT_EYE_LINE = 0.7
_ROLL_WEIGHT_FACE_AXIS = 0.3

# Adaptive scaling.
_LAPLACIAN_SCALE_DIVISOR = 500.0
_LAPLACIAN_CLIP = (0.6, 1.2)


class ImprovedTeleconferenceTracking:
    """Face cropping + head-pose estimation tuned for noisy remote video."""

    def __init__(self) -> None:
        self._face_detector = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=0.7, model_selection=0,
        )

        # Cropping state.
        self.face_roi: tuple[int, int, int, int] | None = None
        self.face_scale = 1.0
        self.roi_padding = TELECONFERENCE_ROI_PADDING

        # Tracking parameters.
        self.scale_factor_base = SCALE_FACTOR_BASE
        self.scale_factor_pitch = SCALE_FACTOR_PITCH
        self.scale_factor_roll = SCALE_FACTOR_ROLL

        self.smoothing_window = SMOOTHING_WINDOW
        self._motion_history = {
            'horizontal': deque(maxlen=self.smoothing_window),
            'vertical': deque(maxlen=self.smoothing_window),
            'roll': deque(maxlen=self.smoothing_window),
        }

        self.deadzone_horizontal = DEADZONE_HORIZONTAL
        self.deadzone_vertical = DEADZONE_VERTICAL
        self.deadzone_roll = DEADZONE_ROLL

    # ---------------------------------------------------------------------
    # Face detection / cropping
    # ---------------------------------------------------------------------

    def detect_and_crop_face(self, frame: np.ndarray) -> tuple[np.ndarray, bool]:
        """Detects a face and returns a target-sized, letterboxed crop.

        Returns:
            Tuple (output_frame, face_detected). When no face is detected, the
            entire input frame is letterboxed to the target size and the flag
            is False.
        """
        if frame is None or frame.size == 0:
            return frame, False

        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_detector.process(rgb)

        if results.detections:
            cropped = self._crop_around_face(frame, results.detections[0],
                                             width, height)
            if cropped is not None:
                return cropped, True

        return self._letterbox(frame, TELECONFERENCE_TARGET_SIZE), False

    def _crop_around_face(self, frame, detection, width, height):
        bbox = detection.location_data.relative_bounding_box
        x = int(bbox.xmin * width)
        y = int(bbox.ymin * height)
        w = int(bbox.width * width)
        h = int(bbox.height * height)

        center_x = x + w // 2
        center_y = y + h // 2
        face_size = int(max(w, h) * self.roi_padding)

        x1 = max(0, center_x - face_size // 2)
        y1 = max(0, center_y - face_size // 2)
        x2 = min(width, center_x + face_size // 2)
        y2 = min(height, center_y + face_size // 2)

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None

        result = self._letterbox(face_crop, TELECONFERENCE_TARGET_SIZE)
        self.face_roi = (x1, y1, x2, y2)
        cropped_h, cropped_w = face_crop.shape[:2]
        self.face_scale = face_size / max(cropped_w, cropped_h)
        return result

    @staticmethod
    def _letterbox(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
        """Resizes `image` to fit `target_size` while preserving aspect ratio."""
        h, w = image.shape[:2]
        target_w, target_h = target_size
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))

        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y_off = (target_h - new_h) // 2
        x_off = (target_w - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        return canvas

    # ---------------------------------------------------------------------
    # Smoothing helpers
    # ---------------------------------------------------------------------

    def _smooth_motion(self, value: float, motion_type: str) -> float:
        history = self._motion_history[motion_type]
        history.append(value)
        if not history:
            return value
        weights = np.exp(np.linspace(0, 1, len(history)))
        weights /= weights.sum()
        return float(np.average(list(history), weights=weights))

    @staticmethod
    def _apply_deadzone(value: float, threshold: float) -> float:
        return 0.0 if abs(value) < threshold else value

    # ---------------------------------------------------------------------
    # Quality-adaptive sensitivity
    # ---------------------------------------------------------------------

    def calculate_adaptive_scale_factors(self, frame: np.ndarray) -> dict:
        """Attenuates sensitivity when the input is blurry."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            clarity = float(np.clip(laplacian_var / _LAPLACIAN_SCALE_DIVISOR,
                                    *_LAPLACIAN_CLIP))
        except Exception:  # noqa: BLE001
            clarity = 1.0
        return {
            'base': self.scale_factor_base * clarity,
            'pitch': self.scale_factor_pitch * clarity,
            'roll': self.scale_factor_roll * clarity,
        }

    # ---------------------------------------------------------------------
    # Pose estimation
    # ---------------------------------------------------------------------

    def improved_head_pose_calculation(self, face_features) -> tuple[float, float, float]:
        """Returns (horizontal_offset, vertical_offset, roll_angle), smoothed."""
        if face_features is None:
            return 0.0, 0.0, 0.0

        nose = face_features[_NOSE_TIP]
        left_eye = face_features[_LEFT_EYE]
        right_eye = face_features[_RIGHT_EYE]
        forehead = face_features[_FOREHEAD]
        chin = face_features[_CHIN]
        mouth = face_features[_MOUTH_CENTER]

        eye_center = (left_eye + right_eye) / 2

        # Horizontal: nose vs eye line.
        h_offset = nose[0] - eye_center[0]
        h_offset = self._apply_deadzone(h_offset, self.deadzone_horizontal)
        h_offset = self._smooth_motion(h_offset, 'horizontal')

        # Vertical: nose vs eye/mouth midline.
        eye_mouth_mid = (eye_center + mouth) / 2
        v_offset = nose[1] - eye_mouth_mid[1]
        v_offset = self._apply_deadzone(v_offset, self.deadzone_vertical)
        v_offset = self._smooth_motion(v_offset, 'vertical')

        # Roll: weighted combination of eye line and face axis.
        eye_line = right_eye - left_eye
        eye_roll = np.arctan2(eye_line[1], eye_line[0])
        face_axis = chin - forehead
        face_roll = np.arctan2(face_axis[0], face_axis[1])
        roll = (_ROLL_WEIGHT_EYE_LINE * eye_roll
                + _ROLL_WEIGHT_FACE_AXIS * face_roll)
        roll = self._apply_deadzone(roll, self.deadzone_roll)
        roll = self._smooth_motion(roll, 'roll')

        return h_offset, v_offset, roll

    @staticmethod
    def map_to_robot_commands(h_offset, v_offset, roll, image_shape, scale_factors):
        """Maps face offsets to base/neck-pitch/neck-roll commands."""
        height, width = image_shape[:2]

        base_position = -np.clip(
            h_offset / width * scale_factors['base'] * MAX_BASE_RANGE,
            -MAX_BASE_RANGE, MAX_BASE_RANGE,
        )
        neck_pitch = -np.clip(
            -v_offset / height * scale_factors['pitch'] * MAX_NECK_PITCH_RANGE,
            -MAX_NECK_PITCH_RANGE, MAX_NECK_PITCH_RANGE,
        )
        neck_roll = np.clip(
            -roll * scale_factors['roll'],
            -MAX_NECK_ROLL_RANGE, MAX_NECK_ROLL_RANGE,
        )
        return base_position, neck_pitch, neck_roll


class EnhancedCalibration:
    """Per-axis baseline calibration using median over the first N frames."""

    def __init__(self, required_frames: int = REQUIRED_CALIBRATION_FRAMES) -> None:
        self.required_frames = required_frames
        self._calibration_data: list[tuple[float, float, float]] = []
        self.is_calibrated = False
        self.baseline: dict[str, float] | None = None

    def add_frame(self, h_offset: float, v_offset: float, roll: float) -> bool:
        """Adds one frame; returns True the moment calibration completes."""
        self._calibration_data.append((h_offset, v_offset, roll))
        if len(self._calibration_data) >= self.required_frames:
            self._compute_baseline()
            return True
        return False

    def _compute_baseline(self) -> None:
        data = np.array(self._calibration_data)
        self.baseline = {
            'horizontal': float(np.median(data[:, 0])),
            'vertical': float(np.median(data[:, 1])),
            'roll': float(np.median(data[:, 2])),
        }
        self.is_calibrated = True
        print(f"Calibrated: h={self.baseline['horizontal']:.3f}, "
              f"v={self.baseline['vertical']:.3f}, "
              f"roll={self.baseline['roll']:.3f}")

    def apply_calibration(self, h_offset: float, v_offset: float,
                          roll: float) -> tuple[float, float, float]:
        if not self.is_calibrated or self.baseline is None:
            return h_offset, v_offset, roll
        return (
            h_offset - self.baseline['horizontal'],
            v_offset - self.baseline['vertical'],
            roll - self.baseline['roll'],
        )
