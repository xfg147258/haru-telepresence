# Expression manager: face-landmark capture, classification, and routine dispatch.

import time

import cv2
import mediapipe as mp
import numpy as np
import torch

from constants import NEUTRAL_EXPRESSION_ID, ROUTINE_IDS
from expression_model import FacialExpressionGCN
from haru2_core_msgs.srv import Routine
from representative_keypoints import (
    get_visualization_points,
    representative_keypoints,
)

_MODEL_WEIGHTS_PATH = 'models/expression_mapping.pth'
_LANDMARK_FRAME_SIZE = (480, 320)
_INPUT_VECTOR_SIZE = 226
_REQUIRED_LANDMARK_DIM = 936
_FRAME_SEQUENCE_LEN = 4
_ROUTINE_SERVICE_TIMEOUT_S = 0.1
_ROUTINE_CALL_TIMEOUT_S = 3.0


class ExpressionManager:
    """Recognizes facial expressions and dispatches matching robot routines."""

    def __init__(self, node, device: torch.device) -> None:
        self.node = node
        self.device = device

        # MediaPipe face mesh.
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self._classifier = self._load_classifier()

        # ROS2 service client for routine execution.
        self._routine_client = self.node.create_client(
            Routine, '/haru2/execute_routine'
        )

        # Landmark sequence buffer (last N frames).
        self.frame_sequence: list[np.ndarray] = []

        # Routine execution state.
        self.executing_routine = False
        self.routine_start_time: float | None = None
        self._pending_future = None
        self._call_time: float | None = None

    # ---------------------------------------------------------------------
    # Model loading
    # ---------------------------------------------------------------------

    def _load_classifier(self) -> FacialExpressionGCN | None:
        """Loads the GCN classifier weights, returning None on failure."""
        try:
            model = FacialExpressionGCN(
                input_size=_INPUT_VECTOR_SIZE,
                num_classes=len(ROUTINE_IDS),
            ).to(self.device)
            model.load_state_dict(
                torch.load(_MODEL_WEIGHTS_PATH, map_location=self.device)
            )
            model.eval()
            self.node.get_logger().info('Expression classifier loaded.')
            return model
        except Exception as exc:  # noqa: BLE001
            self.node.get_logger().error(f'Failed to load classifier: {exc}')
            return None

    # ---------------------------------------------------------------------
    # Landmark extraction
    # ---------------------------------------------------------------------

    def extract_landmarks(self, frame: np.ndarray):
        """Extracts representative + visualization landmarks from a BGR frame.

        Returns:
            Tuple of (rep_keypoints, vis_points). Both are None if no face is
            detected or detection is incomplete.
        """
        small = cv2.resize(frame, _LANDMARK_FRAME_SIZE,
                           interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None, None

        flat = []
        for lm in results.multi_face_landmarks[0].landmark:
            flat.append(lm.x)
            flat.append(lm.y)
        landmarks = np.array(flat)

        if landmarks.shape[0] < _REQUIRED_LANDMARK_DIM:
            return None, None

        return (representative_keypoints(landmarks),
                get_visualization_points(landmarks))

    # ---------------------------------------------------------------------
    # Classification
    # ---------------------------------------------------------------------

    def recognize_expression(self) -> int | None:
        """Runs the classifier on the latest landmarks.

        Returns:
            Routine ID to execute, or None if neutral or classifier
            unavailable.
        """
        if self._classifier is None or len(self.frame_sequence) < _FRAME_SEQUENCE_LEN:
            return None

        latest = self.frame_sequence[-1]
        tensor = (torch.tensor(latest, dtype=torch.float32)
                  .unsqueeze(0).to(self.device))

        with torch.no_grad():
            logits = self._classifier(tensor)
            class_idx = torch.argmax(logits, dim=1).item()
            routine_id = ROUTINE_IDS[class_idx]

        if routine_id == NEUTRAL_EXPRESSION_ID:
            return None
        return routine_id

    # ---------------------------------------------------------------------
    # Routine execution
    # ---------------------------------------------------------------------

    def execute_routine(self, routine_id: int) -> bool:
        """Sends an asynchronous routine request to the robot.

        Returns:
            True if the request was dispatched, False otherwise.
        """
        if self._pending_future is not None:
            return False

        if not self._routine_client.wait_for_service(
                timeout_sec=_ROUTINE_SERVICE_TIMEOUT_S):
            return False

        try:
            request = Routine.Request()
            request.routine = int(routine_id)
            self._pending_future = self._routine_client.call_async(request)
            self._call_time = time.time()
            self._pending_future.add_done_callback(self._on_routine_response)
            return True
        except Exception as exc:  # noqa: BLE001
            self.node.get_logger().error(f'Routine request failed: {exc}')
            self._pending_future = None
            self._call_time = None
            return False

    def _on_routine_response(self, future) -> None:
        """Service-call callback: marks the routine as executing on success."""
        try:
            if future.result() is not None:
                self.executing_routine = True
                self.routine_start_time = time.time()
        except Exception as exc:  # noqa: BLE001
            self.node.get_logger().error(f'Routine service exception: {exc}')
        finally:
            self._pending_future = None
            self._call_time = None

    def check_routine_timeout(self) -> bool:
        """Drops a pending request that has exceeded the call timeout."""
        if (self._pending_future is not None
                and self._call_time is not None
                and time.time() - self._call_time > _ROUTINE_CALL_TIMEOUT_S):
            self._pending_future = None
            self._call_time = None
            return True
        return False

    def is_routine_finished(self, routine_duration: float) -> bool:
        """Returns True the first time the running routine exceeds its duration."""
        if (self.executing_routine
                and self.routine_start_time is not None
                and time.time() - self.routine_start_time > routine_duration):
            self.executing_routine = False
            return True
        return False

    @property
    def has_pending_routine(self) -> bool:
        """True while a routine service call is in flight."""
        return self._pending_future is not None

    # ---------------------------------------------------------------------
    # Frame buffer management
    # ---------------------------------------------------------------------

    def add_frame_to_sequence(self, rep_keypoints: np.ndarray) -> None:
        """Appends a frame; drops the oldest when the buffer is full."""
        self.frame_sequence.append(rep_keypoints)
        if len(self.frame_sequence) > _FRAME_SEQUENCE_LEN:
            self.frame_sequence.pop(0)

    def reset_sequence(self) -> None:
        self.frame_sequence = []
