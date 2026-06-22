"""Face-mesh landmark indices used across the project.

ATTRIBUTION
-----------
The landmark index tables and the 468 -> 113 contour mapping defined in
this module are taken verbatim from the open-source code accompanying:

    Hu, Y., Chen, B., Lin, J., Wang, Y., Wang, Y., Mehlman, C., & Lipson, H.
    (2024). Human-robot facial coexpression. Science Robotics, 9(88),
    eadi4724. https://doi.org/10.1126/scirobotics.adi4724

    Original code release: https://doi.org/10.5061/dryad.gxd2547t7

Specifically, the following symbols are reused unchanged from the
authors' release:
    original_contour_idx, jaw_idx, lips_idx, inner_lips_idx,
    left_eye_idx, right_eye_idx, left_eyebrow_idx, right_eyebrow_idx,
    mesh2contour, and the derived contour_idx / mouth_idx / eyes_idx /
    s_eyes_idx / s_mouth_idx.

The selector functions face_3d_to_2d, mash_to_contour, and
mash_to_contour_half preserve the original logic with only stylistic
edits (type hints, docstrings).

Removed relative to the original release: matplotlib plotting helpers
(plot_2d_face, plot_2d_contour, plot_2d), the connection-edge tables
used only by those plotters, and the file-globbing utilities
(numericalSort, seq_glob). They are not used by this project at runtime.

Please cite the paper above if you build on this project academically.
"""

import numpy as np

# =============================================================================
# Original 128-point face-mesh contour indices
# =============================================================================

original_contour_idx = [
    0, 7, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63,
    65, 66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109,
    127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157,
    158, 159, 160, 161, 162, 163, 172, 173, 176, 178, 181, 185, 191, 234, 246,
    249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295,
    296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334,
    336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382,
    384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454,
    466,
]

# =============================================================================
# Subset indices: jaw, lips, eyes, eyebrows
# =============================================================================

jaw_idx = [
    234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379,
    365, 397, 288, 361, 323, 454,
]
lips_idx = [
    0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61,
    185, 40, 39, 37, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324,
    318, 402, 317, 14, 87, 178, 88, 95,
]
inner_lips_idx = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14,
    87, 178, 88, 95,
]
left_eye_idx = [
    362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381,
    382,
]
right_eye_idx = [
    133, 173, 157, 158, 159, 160, 161, 246, 33, 7, 163, 144, 145, 153, 154, 155,
]
left_eyebrow_idx = [293, 295, 296, 300, 334, 336, 276, 282, 283, 285]
right_eyebrow_idx = [65, 66, 70, 105, 107, 46, 52, 53, 55, 63]

# Combined groups (sorted for stable ordering across runs).
contour_idx = sorted(
    jaw_idx + lips_idx
    + left_eyebrow_idx + right_eyebrow_idx
    + left_eye_idx + right_eye_idx
)
mouth_idx = sorted(jaw_idx + lips_idx)
eyes_idx = sorted(
    left_eyebrow_idx + right_eyebrow_idx
    + left_eye_idx + right_eye_idx
)

# =============================================================================
# Mapping: original 468-mesh index → 113-point contour index
# =============================================================================

mesh2contour = {
    0: 0, 7: 1, 13: 2, 14: 3, 17: 4, 33: 5, 37: 6, 39: 7, 40: 8, 46: 9,
    52: 10, 53: 11, 55: 12, 58: 13, 61: 14, 63: 15, 65: 16, 66: 17, 70: 18,
    78: 19, 80: 20, 81: 21, 82: 22, 84: 23, 87: 24, 88: 25, 91: 26, 93: 27,
    95: 28, 105: 29, 107: 30, 132: 31, 133: 32, 136: 33, 144: 34, 145: 35,
    146: 36, 148: 37, 149: 38, 150: 39, 152: 40, 153: 41, 154: 42, 155: 43,
    157: 44, 158: 45, 159: 46, 160: 47, 161: 48, 163: 49, 172: 50, 173: 51,
    176: 52, 178: 53, 181: 54, 185: 55, 191: 56, 234: 57, 246: 58, 249: 59,
    263: 60, 267: 61, 269: 62, 270: 63, 276: 64, 282: 65, 283: 66, 285: 67,
    288: 68, 291: 69, 293: 70, 295: 71, 296: 72, 300: 73, 308: 74, 310: 75,
    311: 76, 312: 77, 314: 78, 317: 79, 318: 80, 321: 81, 323: 82, 324: 83,
    334: 84, 336: 85, 361: 86, 362: 87, 365: 88, 373: 89, 374: 90, 375: 91,
    377: 92, 378: 93, 379: 94, 380: 95, 381: 96, 382: 97, 384: 98, 385: 99,
    386: 100, 387: 101, 388: 102, 390: 103, 397: 104, 398: 105, 400: 106,
    402: 107, 405: 108, 409: 109, 415: 110, 454: 111, 466: 112,
}

# Pre-mapped subset indices into the 113-point contour space.
s_eyes_idx = [mesh2contour[i] for i in eyes_idx]
s_mouth_idx = [mesh2contour[i] for i in mouth_idx]


# =============================================================================
# Selector utilities (used by data collection / training pipelines)
# =============================================================================

def face_3d_to_2d(landmark: np.ndarray, seq: bool = False) -> np.ndarray:
    """Drops the Z-axis from a 3-D face landmark array.

    Args:
        landmark: Either (3, N) for a single frame or (T, 3, N) for a sequence.
        seq: True if the input is a sequence.
    """
    return landmark[:, :2, :] if seq else landmark[:2, :]


def mash_to_contour(landmark: np.ndarray, seq: bool = False) -> np.ndarray:
    """Selects the 113-point contour subset from a full face-mesh landmark array."""
    return landmark[:, :, contour_idx] if seq else landmark[:, contour_idx]


def mash_to_contour_half(landmark: np.ndarray, seq: bool = False,
                         eyes: bool = False, mouth: bool = False) -> np.ndarray:
    """Selects either the eyes-only or mouth-only contour subset.

    If neither flag is set, returns the full contour subset.
    """
    if eyes:
        half_idx = sorted(
            left_eyebrow_idx + right_eyebrow_idx
            + left_eye_idx + right_eye_idx
        )
    elif mouth:
        half_idx = sorted(jaw_idx + lips_idx)
    else:
        half_idx = contour_idx
    return landmark[:, :, half_idx] if seq else landmark[:, half_idx]
