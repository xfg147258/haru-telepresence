"""LSTM-PINN Vestibulo-Ocular Reflex (VOR) compensator — runtime API.

This module exposes the components that the realtime telepresence pipeline
needs at inference time:

    * Physics constants and the analytical reference simulator
      (VORPhysicsSimulator).
    * The neural network architecture (VORNetLSTM_PINN) used to learn a
      smoother, less-stiff version of the analytical model.
    * The streaming inference wrapper (PINNVORCompensator) that
      eye_tracker.py imports.

Training, validation, plotting, and CLI entry points live in
vor_pinn_train.py; this module deliberately has no training-time
dependencies.

Physics references:
    Fernandez & Goldberg 1971 — semicircular canal time constant.
    Robinson 1977 — velocity storage time constant.
    Guitton & Volle 1987 — maximum eye velocity.
    Carey & Hirvonen 2010 — Ewald excitation/inhibition asymmetry.
    Blanks et al. 1975 — semicircular canal cross-coupling.
"""

import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn

# =============================================================================
# Physics constants
# =============================================================================

TAU_CANAL = 5.7              # Semicircular canal time constant (s).
TAU_VEL_STORAGE = 16.0       # Velocity storage time constant (s).
CANAL_GAIN_BASE = 0.40       # Baseline canal gain.
E_MAX_DEG_S = 500.0          # Maximum eye velocity (deg/s).
SAT_SHARPNESS = 0.015        # softsign sharpness (gamma in Eq. 23).

# Ewald asymmetry (excitation > inhibition).
ASYM_EXCITE = 0.18
ASYM_INHIBIT = 0.10
ASYM_VEL_SCALE = 50.0

# Cross-coupling between axes.
COUPLING_YAW2ROLL = 0.08
COUPLING_PITCH2YAW = 0.05

# Dual-pathway (direct + indirect) blend weights.
K_DIRECT = 0.6
K_INDIRECT = 1.0 - K_DIRECT

# Effective decay time constant of the blended signal after motion stops.
# tau_eff = 1 / (K_DIRECT/tau_c + K_INDIRECT/tau_vs) ≈ 7.7 s.
TAU_BLEND_EFF = 1.0 / (K_DIRECT / TAU_CANAL + K_INDIRECT / TAU_VEL_STORAGE)

DT = 1.0 / 30.0              # Simulation/decode timestep (s); matches ~30 FPS (Eq. 25).

# =============================================================================
# Compensation scaling — values come from constants.py if available.
# =============================================================================

try:
    from constants import (
        MAX_SAFE_EYE_DISPLACEMENT,
        VOR_DEG_TO_PIXEL,
        VOR_HORIZONTAL_COMPENSATION,
        VOR_MIN_VELOCITY_THRESHOLD,
        VOR_OUTPUT_SMOOTHING_ALPHA,
        VOR_ROLL_COMPENSATION,
        VOR_VERTICAL_COMPENSATION,
    )
except ImportError:
    VOR_HORIZONTAL_COMPENSATION = 1.0   # w_y
    VOR_VERTICAL_COMPENSATION = 0.8     # w_p
    VOR_ROLL_COMPENSATION = 0.15        # w_r
    MAX_SAFE_EYE_DISPLACEMENT = 240.0
    VOR_DEG_TO_PIXEL = 1.2              # k_px
    VOR_MIN_VELOCITY_THRESHOLD = 2.0
    VOR_OUTPUT_SMOOTHING_ALPHA = 0.3

MAX_X_PX = MAX_SAFE_EYE_DISPLACEMENT
MAX_Y_PX = MAX_SAFE_EYE_DISPLACEMENT * (
    VOR_VERTICAL_COMPENSATION / VOR_HORIZONTAL_COMPENSATION
)

# =============================================================================
# Default paths
# =============================================================================

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'vor_lstm_pinn.pth')
DEFAULT_PLOT_DIR = os.path.join(os.path.dirname(__file__), 'lstm_results')


# =============================================================================
# Canal-gain helpers (numpy and torch)
# =============================================================================

def canal_gain(omega: np.ndarray) -> np.ndarray:
    """Ewald asymmetric canal gain. Same shape as the input."""
    g = np.full_like(omega, CANAL_GAIN_BASE, dtype=np.float64)
    pos, neg = omega > 0, omega < 0
    g[pos] *= 1.0 + ASYM_EXCITE * np.tanh(omega[pos] / ASYM_VEL_SCALE)
    g[neg] *= 1.0 - ASYM_INHIBIT * np.tanh(-omega[neg] / ASYM_VEL_SCALE)
    return g


def canal_gain_torch(omega: torch.Tensor) -> torch.Tensor:
    """canal_gain implemented with torch ops for batch-mode use in losses."""
    g = torch.full_like(omega, CANAL_GAIN_BASE)
    return torch.where(
        omega > 0,
        g * (1.0 + ASYM_EXCITE * torch.tanh(omega / ASYM_VEL_SCALE)),
        torch.where(
            omega < 0,
            g * (1.0 - ASYM_INHIBIT * torch.tanh(-omega / ASYM_VEL_SCALE)),
            g,
        ),
    )


# =============================================================================
# Analytical (numerical) reference simulator
# =============================================================================

class VORPhysicsSimulator:
    """Discrete-time simulator of the dual-pathway VOR model.

    Used both as a training-time reference target for the neural network and
    as a sanity-check baseline at validation time.
    """

    def __init__(self, dt: float = DT) -> None:
        self.dt = dt
        self.reset()

    def reset(self) -> None:
        self.canal = np.zeros(3, dtype=np.float64)
        self.storage = np.zeros(3, dtype=np.float64)
        self.omega_c_prev = np.zeros(3, dtype=np.float64)

    def _cross_couple(self, omega: np.ndarray) -> np.ndarray:
        c = omega.copy()
        c[2] += COUPLING_YAW2ROLL * omega[0]
        c[0] += COUPLING_PITCH2YAW * omega[1]
        return c

    def step(self, omega_deg: np.ndarray) -> np.ndarray:
        """Advances by one timestep. Returns 2-D (comp_x, comp_y) in pixels."""
        omega = np.asarray(omega_deg, dtype=np.float64)
        omega_c = self._cross_couple(omega)
        gain = canal_gain(omega_c)

        d_omega_c = (omega_c - self.omega_c_prev) / self.dt
        self.canal += (-self.canal / TAU_CANAL + gain * d_omega_c) * self.dt
        self.omega_c_prev = omega_c.copy()

        self.storage += ((-self.storage + self.canal) / TAU_VEL_STORAGE) * self.dt

        blend = K_DIRECT * self.canal + K_INDIRECT * self.storage
        eye_vel = (E_MAX_DEG_S * SAT_SHARPNESS * blend) / (
            1.0 + SAT_SHARPNESS * np.abs(blend))

        # Eq. (29): [-w_y, 0, w_r] / [0, w_p, 0] projection onto the 2-D display.
        # The roll axis contributes only to the horizontal component; torsional
        # eye movement cannot be shown on a flat screen, so it is absent from y.
        comp_x = (
            -eye_vel[0] * VOR_HORIZONTAL_COMPENSATION
            + eye_vel[2] * VOR_ROLL_COMPENSATION
        ) * self.dt * VOR_DEG_TO_PIXEL
        comp_y = (
            eye_vel[1] * VOR_VERTICAL_COMPENSATION
        ) * self.dt * VOR_DEG_TO_PIXEL
        return np.array([comp_x, comp_y], dtype=np.float64)

    def simulate(self, omega_seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Runs `step` over a full trajectory; returns (states, outputs).

        states is a (T, 6) array stacking [canal(3), storage(3)] per step;
        outputs is the (T, 2) compensation in pixels.
        """
        self.reset()
        T = len(omega_seq)
        states = np.zeros((T, 6), dtype=np.float32)
        outputs = np.zeros((T, 2), dtype=np.float32)
        for t in range(T):
            outputs[t] = self.step(omega_seq[t]).astype(np.float32)
            states[t] = np.concatenate([self.canal, self.storage]).astype(np.float32)
        return states, outputs


# =============================================================================
# Neural network architecture
# =============================================================================

class VORNetLSTM_PINN(nn.Module):
    """LSTM that predicts canal and storage state, decoded via the physics model.

    Input  : (B, T, 6) — concatenation of [omega(3), omega_dot(3)] (normalized).
    Output : tuple (comp, c_hat, s_hat, lstm_state)
        comp     — (B, T, 2) pixel compensation per step.
        c_hat    — (B, T, 3) canal state estimate (deg/s).
        s_hat    — (B, T, 3) storage state estimate (deg/s).
        lstm_state — for use in streaming inference.
    """

    DT = DT

    def __init__(self, hidden: int = 64, num_layers: int = 1) -> None:
        super().__init__()
        self.hidden = hidden
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=6, hidden_size=hidden,
            num_layers=num_layers, batch_first=True,
        )
        self.head_canal = nn.Linear(hidden, 3)
        self.head_storage = nn.Linear(hidden, 3)
        self._init_weights()

    def _init_weights(self) -> None:
        for name, p in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
                # Set forget gate bias to 1.
                n = p.size(0)
                p.data[n // 4: n // 2].fill_(1.0)
        for m in (self.head_canal, self.head_storage):
            nn.init.xavier_normal_(m.weight, gain=0.5)
            nn.init.zeros_(m.bias)

    @staticmethod
    def physics_decode_comp(c_hat: torch.Tensor, s_hat: torch.Tensor) -> torch.Tensor:
        """Decodes (canal, storage) state into pixel compensation per step."""
        blend = K_DIRECT * c_hat + K_INDIRECT * s_hat
        eye_vel = (E_MAX_DEG_S * SAT_SHARPNESS * blend) / (
            1.0 + SAT_SHARPNESS * torch.abs(blend))
        # Eq. (29): identical projection to VORPhysicsSimulator.step (roll has
        # no vertical component on a 2-D display).
        comp_x = (
            -eye_vel[..., 0] * VOR_HORIZONTAL_COMPENSATION
            + eye_vel[..., 2] * VOR_ROLL_COMPENSATION
        ) * DT * VOR_DEG_TO_PIXEL
        comp_y = (
            eye_vel[..., 1] * VOR_VERTICAL_COMPENSATION
        ) * DT * VOR_DEG_TO_PIXEL
        return torch.stack([comp_x, comp_y], dim=-1)

    def forward(self, omega: torch.Tensor, state=None):
        h, new_state = self.lstm(omega, state)
        c_hat = self.head_canal(h)
        s_hat = self.head_storage(h)
        comp = self.physics_decode_comp(c_hat, s_hat)
        return comp, c_hat, s_hat, new_state

    def infer_step(self, omega_1step, state):
        """Single-step inference helper used by the streaming compensator."""
        with torch.no_grad():
            x = torch.as_tensor(omega_1step, dtype=torch.float32).view(1, 1, -1)
            comp, _, _, new_state = self.forward(x, state)
        return comp.squeeze().cpu().numpy(), new_state


# =============================================================================
# Streaming inference wrapper
# =============================================================================

_VELOCITY_HISTORY_LEN = 3
_RESET_GAP_S = 0.5
_DT_CLAMP = (0.01, 0.04)
_POSITION_NOISE_FLOOR_RAD = 2e-4


class PINNVORCompensator:
    """Realtime VOR compensator used by EyeTracker.

    Tracks per-axis position history, estimates angular velocity, normalizes
    it for the trained network, runs single-step LSTM inference, and outputs
    a smoothed (delta_x, delta_y) eye-pixel correction every call.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH,
                 device: str = 'cpu') -> None:
        self.device = torch.device(device)
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)

        self.feat_mean = ckpt['feat_mean']
        self.feat_std = ckpt['feat_std']
        self.feat_mean_dot = ckpt['feat_mean_dot']
        self.feat_std_dot = ckpt['feat_std_dot']
        self.tgt_std = ckpt['tgt_std']

        config = ckpt['config']
        self.model = VORNetLSTM_PINN(
            hidden=config.get('hidden', 64),
            num_layers=config.get('num_layers', 1),
        )
        self.model.load_state_dict(ckpt['model_state'])
        self.model.eval()

        self._lstm_state = None
        self.prev_time: float | None = None
        self._prev_omega = np.zeros(3, dtype=np.float32)

        self.min_velocity = VOR_MIN_VELOCITY_THRESHOLD
        self._alpha = VOR_OUTPUT_SMOOTHING_ALPHA
        self.smoothed = {'x': 0.0, 'y': 0.0}

        axes = ('base', 'pitch', 'roll')
        self._pos_hist = {ax: deque(maxlen=_VELOCITY_HISTORY_LEN) for ax in axes}
        self._time_hist = {ax: deque(maxlen=_VELOCITY_HISTORY_LEN) for ax in axes}
        self._last_velocities = {ax: 0.0 for ax in axes}
        self.smoothed_output = {ax: 0.0 for ax in axes}

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def compute_vor_compensation(self, base_position: float, neck_pitch: float,
                                 neck_roll: float) -> tuple[float, float]:
        """Returns (delta_x, delta_y) eye-pixel compensation for this frame."""
        now = time.time()
        if self.prev_time is None:
            self.prev_time = now
            return 0.0, 0.0

        dt = float(np.clip(now - self.prev_time, *_DT_CLAMP))
        self.prev_time = now

        # Long pause → reset everything; large dt blows up dω/dt otherwise.
        if dt > _RESET_GAP_S:
            self._reset_runtime_state()
            return 0.0, 0.0

        omega = np.array([
            self._calc_velocity(base_position, 'base'),
            self._calc_velocity(neck_pitch, 'pitch'),
            self._calc_velocity(neck_roll, 'roll'),
        ], dtype=np.float32)

        # Below threshold: relax smoothed output toward zero, no model call.
        if np.abs(omega).sum() < self.min_velocity:
            self.smoothed['x'] *= (1 - self._alpha)
            self.smoothed['y'] *= (1 - self._alpha)
            return 0.0, 0.0

        comp_x, comp_y = self._predict(omega)

        self.smoothed['x'] = self._alpha * comp_x + (1 - self._alpha) * self.smoothed['x']
        self.smoothed['y'] = self._alpha * comp_y + (1 - self._alpha) * self.smoothed['y']

        self.smoothed_output['base'] = self.smoothed['x']
        self.smoothed_output['pitch'] = self.smoothed['y']
        self.smoothed_output['roll'] = 0.0

        return (
            float(np.clip(self.smoothed['x'], -MAX_X_PX, MAX_X_PX)),
            float(np.clip(self.smoothed['y'], -MAX_Y_PX, MAX_Y_PX)),
        )

    def reset(self) -> None:
        self._reset_runtime_state()

    def get_velocity_info(self) -> dict:
        return dict(self._last_velocities)

    def get_internal_states(self) -> dict:
        """Returns LSTM hidden state as a yaw/pitch/roll-named diagnostic dict."""
        canal = {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}
        if self._lstm_state is not None:
            try:
                h = self._lstm_state[0]
                h_last = h[-1, 0, :].detach().cpu().numpy()
                canal = {
                    'yaw': float(h_last[0]),
                    'pitch': float(h_last[1]),
                    'roll': float(h_last[2]),
                }
            except Exception:  # noqa: BLE001
                pass
        return {
            'pinn_ok': True,
            'canal': canal,
            'storage': {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0},
        }

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------

    def _reset_runtime_state(self) -> None:
        self._lstm_state = None
        for ax in ('base', 'pitch', 'roll'):
            self._pos_hist[ax].clear()
            self._time_hist[ax].clear()
            self._last_velocities[ax] = 0.0
            self.smoothed_output[ax] = 0.0
        self.smoothed = {'x': 0.0, 'y': 0.0}
        self._prev_omega = np.zeros(3, dtype=np.float32)
        self.prev_time = None

    def _calc_velocity(self, position: float, axis: str) -> float:
        """Estimates instantaneous angular velocity in deg/s for one axis."""
        now = time.time()
        self._pos_hist[axis].append(position)
        self._time_hist[axis].append(now)

        ph = list(self._pos_hist[axis])
        th = list(self._time_hist[axis])
        if len(ph) < 2:
            return 0.0
        if abs(ph[-1] - ph[-2]) < _POSITION_NOISE_FLOOR_RAD:
            return 0.0

        # Prefer a 3-tap central difference when available.
        if len(ph) >= 3:
            dt = th[-1] - th[-3]
            v = (ph[-1] - ph[-3]) / dt if dt > 1e-6 else 0.0
        else:
            dt = th[-1] - th[-2]
            v = (ph[-1] - ph[-2]) / dt if dt > 1e-6 else 0.0

        velocity_deg = float(np.degrees(v))
        self._last_velocities[axis] = velocity_deg
        return velocity_deg

    def _predict(self, omega: np.ndarray) -> tuple[float, float]:
        """One LSTM inference step. Returns raw (comp_x, comp_y) in pixels."""
        omega_n = (omega - self.feat_mean) / self.feat_std
        omega_dot = (omega - self._prev_omega) / DT
        omega_dot_n = (omega_dot - self.feat_mean_dot) / self.feat_std_dot
        self._prev_omega = omega.copy()

        inp = np.concatenate([omega_n, omega_dot_n]).astype(np.float32)
        comp, self._lstm_state = self.model.infer_step(inp, self._lstm_state)
        return float(comp[0]), float(comp[1])
