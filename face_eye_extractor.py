"""Face landmark and iris extraction with simulated gaze micromovements.

The module exposes two classes:

* GazeMicromovementSimulator — simulates the three components of human
  fixational eye movement (slow drift, tremor, and microsaccades) and
  returns a small per-frame offset that can be added to the iris-derived
  gaze target to break perfectly static gaze.

* FaceExtractor — wraps MediaPipe FaceMesh with refine_landmarks=True,
  exposes a 478-point face/iris landmark array per frame, and computes
  iris-relative gaze offsets that already include the micromovement
  contribution.
"""

import math
import random
import time

import cv2
import mediapipe as mp
import numpy as np

# =============================================================================
# Gaze micromovement simulator
# =============================================================================

# Drift parameters (slow random walk of the fixation point).
_DRIFT_STEP_STD_PX = 0.3             # Per-frame Gaussian increment std (px).

# Tremor parameters (high-frequency tremor superimposed on drift).
_TREMOR_AMPLITUDE_PX = 0.2
_TREMOR_FREQ_HZ = 80.0

# Microsaccade parameters.
_MICROSACCADE_AMPLITUDE_PX = 2.0
_MICROSACCADE_TRIGGER_DRIFT_PX = 3.0
_MICROSACCADE_DURATION_S = 0.02
_RETINA_ADAPTATION_S = 2.0


class GazeMicromovementSimulator:
    """Simulates fixational drift, tremor, and microsaccades."""

    def __init__(self) -> None:
        self.drift_step_std = _DRIFT_STEP_STD_PX

        self.tremor_amplitude = _TREMOR_AMPLITUDE_PX
        self.tremor_frequency = _TREMOR_FREQ_HZ

        self.microsaccade_amplitude = _MICROSACCADE_AMPLITUDE_PX
        self.microsaccade_threshold = _MICROSACCADE_TRIGGER_DRIFT_PX
        self.microsaccade_duration = _MICROSACCADE_DURATION_S

        self.drift_position = np.array([0.0, 0.0])

        self.microsaccade_active = False
        self.microsaccade_start_time = 0.0
        self.microsaccade_start_position = np.array([0.0, 0.0])
        self.microsaccade_target_position = np.array([0.0, 0.0])

        self.total_offset = np.array([0.0, 0.0])
        self.last_update_time = time.time()

        self.retina_adaptation_time = _RETINA_ADAPTATION_S
        self.last_saccade_time = time.time()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def update(self, current_time: float) -> np.ndarray:
        """Advances all three micromovement components and returns the offset."""
        self._update_drift()
        self._update_microsaccade(current_time)
        self.total_offset = self.drift_position + self._get_tremor_offset(current_time)

        if not self.microsaccade_active:
            self._maybe_trigger_microsaccade(current_time)

        self.last_update_time = current_time
        return self.total_offset

    def reset(self) -> None:
        self.drift_position = np.array([0.0, 0.0])
        self.microsaccade_active = False
        self.total_offset = np.array([0.0, 0.0])

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _random_unit_vector() -> np.ndarray:
        v = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
        return v / np.linalg.norm(v)

    def _update_drift(self) -> None:
        """Eq. (33): drift as a slow 2-D random walk of small Gaussian steps.

        No hard cap is applied: the corrective microsaccade (directed opposite
        to the accumulated drift) is what keeps the fixation point bounded.
        """
        self.drift_position = self.drift_position + np.random.normal(
            0.0, self.drift_step_std, size=2)

    def _get_tremor_offset(self, current_time: float) -> np.ndarray:
        """Multi-frequency tremor: dominant sine plus a higher-harmonic mix."""
        phase = 2 * math.pi * self.tremor_frequency * current_time
        a = self.tremor_amplitude
        offset_x = (a * math.sin(phase)
                    + 0.3 * a * math.sin(2.3 * phase + 1.2)
                    + 0.1 * a * random.uniform(-1, 1))
        offset_y = (a * math.cos(phase + 0.5)
                    + 0.3 * a * math.cos(1.7 * phase + 0.8)
                    + 0.1 * a * random.uniform(-1, 1))
        return np.array([offset_x, offset_y])

    def _maybe_trigger_microsaccade(self, current_time: float) -> None:
        if (np.linalg.norm(self.drift_position) > self.microsaccade_threshold
                or current_time - self.last_saccade_time > self.retina_adaptation_time):
            self._trigger_microsaccade(current_time)

    def _trigger_microsaccade(self, current_time: float) -> None:
        """Corrective jump directed opposite to the accumulated drift vector."""
        drift_norm = np.linalg.norm(self.drift_position)
        if drift_norm > 1e-6:
            direction = -self.drift_position / drift_norm
        else:
            direction = self._random_unit_vector()
        distance = random.uniform(0.5, self.microsaccade_amplitude)

        self.microsaccade_start_position = self.drift_position.copy()
        self.microsaccade_target_position = direction * distance
        self.microsaccade_active = True
        self.microsaccade_start_time = current_time
        self.last_saccade_time = current_time

    def _update_microsaccade(self, current_time: float) -> None:
        if not self.microsaccade_active:
            return
        elapsed = current_time - self.microsaccade_start_time
        if elapsed >= self.microsaccade_duration:
            self.microsaccade_active = False
            self.drift_position = self.microsaccade_target_position
            return

        progress = elapsed / self.microsaccade_duration
        # Sigmoid-like easing: slow at start and end, fast in the middle.
        t = 1 / (1 + math.exp(-10 * (progress - 0.5)))
        self.drift_position = (
            self.microsaccade_start_position * (1 - t)
            + self.microsaccade_target_position * t
        )


# =============================================================================
# Face / iris extractor
# =============================================================================

# MediaPipe face-mesh indices.
_KEY_LANDMARKS = {
    'nose_tip': 1,
    'left_eye_center': 133,
    'right_eye_center': 362,
    'left_eye_inner': 133,
    'left_eye_outer': 33,
    'right_eye_inner': 362,
    'right_eye_outer': 263,
    'mouth_center': 13,
    'forehead_center': 9,
    'chin': 175,
}
_IRIS_LANDMARKS = {
    'left_iris_center': 468,
    'right_iris_center': 473,
}
_EYE_CORNERS = {
    'left_eye_left': 33,
    'left_eye_right': 133,
    'right_eye_left': 362,
    'right_eye_right': 263,
}
# Upper/lower eyelid landmarks used for the eye height of Eq. (19). Verify
# these indices against your MediaPipe FaceMesh (refine_landmarks=True) layout.
_EYE_LIDS = {
    'left_eye_top': 159,
    'left_eye_bottom': 145,
    'right_eye_top': 386,
    'right_eye_bottom': 374,
}

_REQUIRED_LANDMARK_COUNT = 478
_FACE_MESH_DETECTION_CONFIDENCE = 0.7
_FACE_MESH_TRACKING_CONFIDENCE = 0.5


def _normalize_iris_offset(offset_px: np.ndarray,
                           eye_width: float, eye_height: float) -> np.ndarray:
    """Eq. (18)-(19): divide the x-offset by eye width and the y-offset by eye height."""
    w = eye_width if eye_width > 1e-6 else 1.0
    h = eye_height if eye_height > 1e-6 else 1.0
    return np.array([offset_px[0] / w, offset_px[1] / h])


def _denormalize_iris_offset(offset_norm: np.ndarray,
                             eye_width: float, eye_height: float) -> np.ndarray:
    """Inverse of _normalize_iris_offset (used for pixel-space visualization)."""
    return np.array([offset_norm[0] * eye_width, offset_norm[1] * eye_height])


class FaceExtractor:
    """MediaPipe face mesh + iris extractor with simulated gaze micromovements."""

    def __init__(self) -> None:
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=_FACE_MESH_DETECTION_CONFIDENCE,
            min_tracking_confidence=_FACE_MESH_TRACKING_CONFIDENCE,
        )
        self.key_landmarks = _KEY_LANDMARKS
        self.iris_landmarks = _IRIS_LANDMARKS
        self.eye_corners = _EYE_CORNERS
        self.eye_lids = _EYE_LIDS
        self.gaze_micromovement = GazeMicromovementSimulator()
        self.last_frame_time = time.time()

    # ---------------------------------------------------------------------
    # Landmark extraction
    # ---------------------------------------------------------------------

    def extract(self, frame: np.ndarray) -> np.ndarray | None:
        """Returns 478×3 face/iris landmarks in pixel coordinates, or None."""
        if frame is None:
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None

        height, width, _ = frame.shape
        landmarks = results.multi_face_landmarks[0]
        return np.array(
            [[lm.x * width, lm.y * height, lm.z] for lm in landmarks.landmark],
            dtype=np.float32,
        )

    # ---------------------------------------------------------------------
    # Iris / gaze
    # ---------------------------------------------------------------------

    def get_iris_data(self, face_points: np.ndarray | None,
                      current_time: float | None = None) -> dict | None:
        """Computes iris offsets and adds simulated micromovement.

        Returns a dict with the following entries:
            * Eye corners and centers (px).
            * Eye widths.
            * Raw iris offsets (px and normalized to eye width).
            * Smoothed iris offsets including simulated micromovement.
            * Average offsets (mean of left+right).
            * Micromovement offset (px) and microsaccade-active flag.
            * Gaze angles in radians.
        """
        if face_points is None or len(face_points) < _REQUIRED_LANDMARK_COUNT:
            return None
        if current_time is None:
            current_time = time.time()

        data = {}
        self._populate_eye_corners(data, face_points)
        self._populate_iris_centers(data, face_points)
        self._populate_iris_offsets(data)

        # Add simulated fixational micromovement.
        micromovement = self.gaze_micromovement.update(current_time)
        self._apply_micromovement(data, micromovement)

        data['micromovement_offset'] = micromovement
        data['micromovement_active'] = self.gaze_micromovement.microsaccade_active

        avg = data['avg_iris_offset']
        data['gaze_angle_horizontal'] = float(np.arctan2(avg[0], avg[1]))
        data['gaze_angle_vertical'] = float(np.arctan2(avg[1], avg[0]))

        self.last_frame_time = current_time
        return data

not a git repository (or any of the parent directories): .git


    def _populate_iris_centers(self, data: dict, face_points: np.ndarray) -> None:
        data['left_iris_center'] = face_points[self.iris_landmarks['left_iris_center']]
        data['right_iris_center'] = face_points[self.iris_landmarks['right_iris_center']]

    @staticmethod
    def _populate_iris_offsets(data: dict) -> None:
        data['left_iris_offset_raw'] = (
            data['left_iris_center'][:2] - data['left_eye_center'][:2])
        data['right_iris_offset_raw'] = (
            data['right_iris_center'][:2] - data['right_eye_center'][:2])

        # Eq. (18)-(19): normalize x by eye width and y by eye height, so a
        # full-range iris maps to roughly [-0.5, 0.5] on each axis.
        data['left_iris_offset_norm_raw'] = _normalize_iris_offset(
            data['left_iris_offset_raw'],
            data['left_eye_width'], data['left_eye_height'])
        data['right_iris_offset_norm_raw'] = _normalize_iris_offset(
            data['right_iris_offset_raw'],
            data['right_eye_width'], data['right_eye_height'])

    @staticmethod
    def _apply_micromovement(data: dict, micromovement: np.ndarray) -> None:
        # The micromovement is a pixel offset; normalize it with the same
        # anisotropic scaling as the iris offset before adding (Eq. 18-19).
        micro_left = _normalize_iris_offset(
            micromovement, data['left_eye_width'], data['left_eye_height'])
        micro_right = _normalize_iris_offset(
            micromovement, data['right_eye_width'], data['right_eye_height'])
        data['left_iris_offset_norm'] = data['left_iris_offset_norm_raw'] + micro_left
        data['right_iris_offset_norm'] = data['right_iris_offset_norm_raw'] + micro_right

        data['left_iris_offset'] = _denormalize_iris_offset(
            data['left_iris_offset_norm'],
            data['left_eye_width'], data['left_eye_height'])
        data['right_iris_offset'] = _denormalize_iris_offset(
            data['right_iris_offset_norm'],
            data['right_eye_width'], data['right_eye_height'])

        data['avg_iris_offset'] = (
            data['left_iris_offset'] + data['right_iris_offset']) / 2
        data['avg_iris_offset_norm'] = (
            data['left_iris_offset_norm'] + data['right_iris_offset_norm']) / 2
        data['avg_iris_offset_raw'] = (
            data['left_iris_offset_raw'] + data['right_iris_offset_raw']) / 2

    # ---------------------------------------------------------------------
    # Convenience helpers
    # ---------------------------------------------------------------------

    def get_key_landmarks(self, face_points) -> dict | None:
        if face_points is None:
            return None
        return {
            name: face_points[idx]
            for name, idx in self.key_landmarks.items()
            if idx < len(face_points)
        }

    def calculate_face_orientation(self, face_points) -> tuple[float, float, float]:
        """Returns rough (pitch_offset, yaw_offset, roll_angle)."""
        if face_points is None:
            return 0.0, 0.0, 0.0

        nose = face_points[self.key_landmarks['nose_tip']]
        left_eye = face_points[self.key_landmarks['left_eye_center']]
        right_eye = face_points[self.key_landmarks['right_eye_center']]

        eye_center = (left_eye + right_eye) / 2
        yaw = float(nose[0] - eye_center[0])
        pitch = float(nose[1] - eye_center[1])

        eye_vec = right_eye - left_eye
        roll = float(np.arctan2(eye_vec[1], eye_vec[0]))
        return pitch, yaw, roll

    def is_face_stable(self, face_points, previous_points,
                       threshold: float = 10.0) -> bool:
        if face_points is None or previous_points is None:
            return False

        movement_total = 0.0
        for idx in (self.key_landmarks['nose_tip'],
                    self.key_landmarks['left_eye_center'],
                    self.key_landmarks['right_eye_center']):
            if idx < len(face_points) and idx < len(previous_points):
                movement_total += float(np.linalg.norm(
                    face_points[idx][:2] - previous_points[idx][:2]))
        return movement_total < threshold

    # ---------------------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------------------

    def draw_landmarks(self, frame: np.ndarray, face_points,
                       draw_all: bool = False, draw_iris: bool = True,
                       draw_micromovement: bool = True) -> np.ndarray:
        """Annotates the frame with key points, iris centers, and gaze info."""
        if face_points is None:
            return frame

        out = frame.copy()
        if not draw_all:
            self._draw_key_points(out, face_points)
        if draw_iris and len(face_points) >= _REQUIRED_LANDMARK_COUNT:
            self._draw_iris_overlay(out, face_points, draw_micromovement)
        return out

    def _draw_key_points(self, frame, face_points) -> None:
        keys = self.get_key_landmarks(face_points)
        if not keys:
            return

        nose = keys['nose_tip']
        cv2.circle(frame, (int(nose[0]), int(nose[1])), 5, (255, 255, 0), -1)

        for eye_key in ('left_eye_center', 'right_eye_center'):
            eye = keys[eye_key]
            cv2.circle(frame, (int(eye[0]), int(eye[1])), 3, (0, 255, 0), -1)

        left = keys['left_eye_center']
        right = keys['right_eye_center']
        cv2.line(frame,
                 (int(left[0]), int(left[1])), (int(right[0]), int(right[1])),
                 (0, 255, 255), 2)

        mouth = keys['mouth_center']
        cv2.circle(frame, (int(mouth[0]), int(mouth[1])), 2, (255, 0, 0), -1)

    def _draw_iris_overlay(self, frame, face_points,
                           draw_micromovement: bool) -> None:
        iris_data = self.get_iris_data(face_points, time.time())
        if not iris_data:
            return

        for iris_key, color in (('left_iris_center', (255, 255, 0)),
                                ('right_iris_center', (255, 0, 255))):
            p = iris_data[iris_key]
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, color, -1)

        for eye_key, iris_key in (('left_eye_center', 'left_iris_center'),
                                  ('right_eye_center', 'right_iris_center')):
            e = iris_data[eye_key]
            i = iris_data[iris_key]
            cv2.line(frame,
                     (int(e[0]), int(e[1])), (int(i[0]), int(i[1])),
                     (0, 255, 255), 1)

        ox, oy = iris_data['avg_iris_offset_raw']
        cv2.putText(frame, f'Raw Offset: ({ox:.1f}, {oy:.1f})',
                    (10, frame.shape[0] - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if not draw_micromovement:
            return

        mx, my = iris_data['micromovement_offset']
        cv2.putText(frame, f'Micro: ({mx:.2f}, {my:.2f})',
                    (10, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 100), 1)
        if iris_data['micromovement_active']:
            cv2.putText(frame, 'MICROSACCADE!',
                        (10, frame.shape[0] - 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
