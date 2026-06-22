"""System-wide constants for the Haru 2.0 telepresence system.

This module groups all tunable constants by subsystem: routine IDs, motor
ranges and IDs, tracking parameters, calibration, expression recognition,
VOR (vestibulo-ocular reflex), eye mapping, audio, and teleconference mode.
"""

# =============================================================================
# Routine IDs
# =============================================================================

# All routines that the inverse model can output.
ROUTINE_IDS = [
    2092, 2083, 2068, 2057, 2006, 2037,
    2036, 2087, 2010, 2071, 2073, 2023,
    2081, 2021, 2067, 2059, 2017,
]

# Routine treated as "neutral" at runtime (no robot action triggered).
NEUTRAL_EXPRESSION_ID = 2023

# =============================================================================
# Robot motor ranges
# =============================================================================

MAX_BASE_RANGE = 1.57          # Base yaw range, ±90° in radians.
MAX_NECK_PITCH_RANGE = 0.6     # Neck pitch range in radians.
MAX_NECK_ROLL_RANGE = 0.3      # Neck roll range in radians.

MAX_EYE_X = 720.0              # Eye horizontal range in pixels.
MAX_EYE_Y = 720.0              # Eye vertical range in pixels.
CENTER_EYE_X = MAX_EYE_X / 2
CENTER_EYE_Y = MAX_EYE_Y / 2

# =============================================================================
# Motor IDs
# =============================================================================

MOTOR_IDS = {
    'base_yaw': 10,
    'neck_pitch': 11,
    'neck_roll': 12,
}

# =============================================================================
# Head tracking
# =============================================================================

# Sensitivity multipliers (camera offset → motor command).
SCALE_FACTOR_BASE = 9.0
SCALE_FACTOR_PITCH = 12.0
SCALE_FACTOR_ROLL = 4.0

# Motion smoothing window (frames).
SMOOTHING_WINDOW = 7

# Deadzone thresholds (normalized offsets).
DEADZONE_HORIZONTAL = 0.02
DEADZONE_VERTICAL = 0.02
DEADZONE_ROLL = 0.05

# =============================================================================
# Calibration
# =============================================================================

REQUIRED_CALIBRATION_FRAMES = 30
TELECONFERENCE_CALIBRATION_FRAMES = 50

# =============================================================================
# Expression recognition
# =============================================================================

SAMPLE_INTERVAL = 0.1          # Frame sampling interval (s).
ACTION_DURATION = 2.0          # Action duration (s).
ROUTINE_DURATION = 1.3         # Routine execution time (s).
STABLE_THRESHOLD = 1.0         # Stability threshold (s).
MOVEMENT_THRESHOLD = 10.0      # Movement threshold (px).
MAX_HEAD_ROTATION_ANGLE = 15.0 # Max head rotation allowed during recognition.

# =============================================================================
# VOR (Vestibulo-Ocular Reflex)
#
# The VOR is modeled with a nonlinear, physics-guided LSTM (PG-LSTM) over a
# dual-pathway semicircular-canal / velocity-storage ODE (see vor_pinn.py),
# NOT a linear transfer function. The constants below only configure the
# realtime wrapper and the fixed screen-space projection of Eq. (29) that maps
# the decoded 3-axis eye velocity (yaw, pitch, roll) to a 2-D pixel offset.
# =============================================================================

VOR_MIN_VELOCITY_THRESHOLD = 2.0       # Min angular velocity to run the model (deg/s).
VOR_OUTPUT_SMOOTHING_ALPHA = 0.3       # Output EMA smoothing factor.

# Per-axis projection weights of Eq. (29): w_y (yaw), w_p (pitch), w_r (roll).
# Only the secondary horizontal component of the roll axis is kept (w_r),
# because torsional eye movement cannot be represented on a flat display.
VOR_HORIZONTAL_COMPENSATION = 1.0      # w_y
VOR_VERTICAL_COMPENSATION = 0.8        # w_p
VOR_ROLL_COMPENSATION = 0.15           # w_r

# Display calibration constant k_px of Eq. (29) (deg/s · s → pixels).
VOR_DEG_TO_PIXEL = 1.2                 # k_px

# =============================================================================
# Eye mapping
# =============================================================================

EYE_SCALE_FACTOR = 720.0               # Iris offset → robot eye pixel scale.
IRIS_OFFSET_SMOOTHING_WINDOW = 5

EYE_SCREEN_SIZE = 720.0                # Robot eye screen resolution (px).
PUPIL_DIAMETER_PX = EYE_SCREEN_SIZE / 3.0
PUPIL_RADIUS_PX = PUPIL_DIAMETER_PX / 2.0
# Maximum displacement that keeps the pupil fully inside the screen.
MAX_SAFE_EYE_DISPLACEMENT = (EYE_SCREEN_SIZE / 2.0) - PUPIL_RADIUS_PX

# =============================================================================
# Audio / speech recognition
# =============================================================================

AUDIO_CHUNK = 480
AUDIO_FORMAT = 'paInt16'
AUDIO_CHANNELS = 1
AUDIO_RATE = 16000

# Realtime recognition timing (s).
MIN_SPEECH_LENGTH = 0.5
SILENCE_TIMEOUT = 1.2
PROCESS_INTERVAL = 0.3
SPEECH_TIMEOUT = 3.0

# Remote audio level thresholds.
SPEECH_THRESHOLD = 16.0
SILENCE_THRESHOLD = 13.0
MAX_AUDIO_ERRORS = 10

# =============================================================================
# Teleconference mode
# =============================================================================

TELECONFERENCE_ROI_PADDING = 1.8
TELECONFERENCE_TARGET_SIZE = (640, 480)
