"""Eye tracker combining iris-based gaze with PINN VOR compensation."""

from collections import deque

import numpy as np

from constants import (
    CENTER_EYE_X,
    CENTER_EYE_Y,
    EYE_SCALE_FACTOR,
    IRIS_OFFSET_SMOOTHING_WINDOW,
    MAX_EYE_X,
    MAX_EYE_Y,
    VOR_MIN_VELOCITY_THRESHOLD,
)
from vor_pinn import PINNVORCompensator

_HEAD_MOTION_KEYPOINTS = (1, 133, 362)   # Nose tip, inner corners of eyes.
_HEAD_MOTION_THRESHOLD = 5.0             # Pixels averaged across keypoints.


class EyeTracker:
    """Maps iris offsets to robot eye coordinates and applies VOR compensation."""

    def __init__(self) -> None:
        self.eye_scale_factor = EYE_SCALE_FACTOR
        self._iris_history = deque(maxlen=IRIS_OFFSET_SMOOTHING_WINDOW)
        self._vor_compensator = PINNVORCompensator()

        self.head_is_moving = False
        self._prev_head_keypoints: list | None = None

    # ---------------------------------------------------------------------
    # Head motion detection
    # ---------------------------------------------------------------------

    def detect_head_motion(self, face_features) -> tuple[bool, float]:
        """Estimates head motion from a small set of stable landmarks.

        Returns:
            Tuple (is_moving, normalized_motion_score in [0, 1]).
        """
        if face_features is None:
            return False, 0.0

        keypoints = [face_features[i] for i in _HEAD_MOTION_KEYPOINTS]

        if self._prev_head_keypoints is None:
            self._prev_head_keypoints = keypoints
            return False, 0.0

        movement = sum(
            np.linalg.norm(p[:2] - q[:2])
            for p, q in zip(keypoints, self._prev_head_keypoints)
        )
        avg_movement = movement / len(keypoints)
        self._prev_head_keypoints = keypoints
        self.head_is_moving = avg_movement > _HEAD_MOTION_THRESHOLD
        return self.head_is_moving, float(np.tanh(avg_movement / 20.0))

    # ---------------------------------------------------------------------
    # Eye position
    # ---------------------------------------------------------------------

    def calculate_eye_position(self, face_features, iris_data,
                               base_position, neck_pitch, neck_roll):
        """Combines iris-based gaze with PINN VOR compensation.

        Args:
            face_features: MediaPipe face landmarks (unused here, kept for API
                compatibility with potential future variants).
            iris_data: Dict from FaceEyeExtractor.get_iris_data, or None.
            base_position: Current base yaw (rad).
            neck_pitch: Current neck pitch (rad).
            neck_roll: Current neck roll (rad).

        Returns:
            (left_eye_x, left_eye_y, right_eye_x, right_eye_y), all in pixels.
            Both eyes get the same target.
        """
        if iris_data is None:
            return CENTER_EYE_X, CENTER_EYE_Y, CENTER_EYE_X, CENTER_EYE_Y

        # Smooth iris offset.
        self._iris_history.append(iris_data['avg_iris_offset_norm'])
        smoothed = np.mean(list(self._iris_history), axis=0)

        # Map to robot eye coordinates.
        eye_x = CENTER_EYE_X + smoothed[0] * self.eye_scale_factor
        eye_y = CENTER_EYE_Y + smoothed[1] * self.eye_scale_factor

        # VOR compensation (already in pixels-per-frame).
        vor_x, vor_y = self._vor_compensator.compute_vor_compensation(
            base_position, neck_pitch, neck_roll
        )
        eye_x += vor_x
        eye_y += vor_y

        eye_x = float(np.clip(eye_x, 0, MAX_EYE_X))
        eye_y = float(np.clip(eye_y, 0, MAX_EYE_Y))
        return eye_x, eye_y, eye_x, eye_y

    # ---------------------------------------------------------------------
    # Status
    # ---------------------------------------------------------------------

    def get_vor_status(self) -> dict:
        """Returns a snapshot of VOR state for visualization / logging."""
        velocities = self._vor_compensator.get_velocity_info()
        total_velocity = sum(abs(v) for v in velocities.values())

        status = {
            'velocities': velocities,
            'compensations': self._vor_compensator.smoothed_output,
            'is_active': total_velocity > VOR_MIN_VELOCITY_THRESHOLD,
            'total_velocity': total_velocity,
        }

        if hasattr(self._vor_compensator, 'get_internal_states'):
            status['internal_states'] = self._vor_compensator.get_internal_states()
        return status
