"""Helpers for selecting and normalizing the subset of face landmarks used by
the expression classifier."""

import numpy as np

from simplified_landmark_plus import s_eyes_idx, s_mouth_idx

_NORMALIZE_EPS = 1e-6


def representative_keypoints(landmarks_936: np.ndarray,
                             normalize: bool = True) -> np.ndarray:
    """Selects eye+mouth landmarks and (optionally) normalizes them.

    Args:
        landmarks_936: Flat array of 468 face-mesh landmarks as (x, y) pairs,
            length 936.
        normalize: If True, center on the face centroid and scale by the
            largest per-axis standard deviation.

    Returns:
        Flattened 1-D array of selected (and possibly normalized) keypoints.
    """
    points = landmarks_936.reshape(-1, 2)
    rep_points = np.concatenate(
        (points[s_eyes_idx], points[s_mouth_idx]), axis=0
    )

    if not normalize:
        return rep_points.flatten()

    center = np.mean(rep_points, axis=0)
    scale = np.max(np.std(rep_points, axis=0))
    if scale < _NORMALIZE_EPS:
        scale = 1.0
    return ((rep_points - center) / scale).flatten()


def get_visualization_points(landmarks_936: np.ndarray) -> np.ndarray:
    """Returns the same eye+mouth subset, but un-normalized as 2-D points.

    Args:
        landmarks_936: Flat array of 468 face-mesh landmarks.

    Returns:
        Array of shape (N, 2) with original-coordinate keypoints.
    """
    points = landmarks_936.reshape(-1, 2)
    return np.concatenate(
        (points[s_eyes_idx], points[s_mouth_idx]), axis=0
    )
