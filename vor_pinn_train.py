""" LSTM-PINN VOR Training and Validation

Core Components:
- Training loss: L = L_data + lambda_s * L_state  (Eq. 30)
- Data scenarios: sinusoidal, pseudorandom, constant velocity, velocity step
  with post-rotatory decay, head impulses, and three-axis coupled rotation.
- Validation Metrics: Step response RMSE, Sine tracking RMSE + phase lag
- Plots: Training loss curve + validation sine & decay plots
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False

from vor_pinn import (
    CANAL_GAIN_BASE,
    DEFAULT_MODEL_PATH,
    DEFAULT_PLOT_DIR,
    DT,
    MAX_X_PX,
    MAX_Y_PX,
    TAU_CANAL,
    TAU_VEL_STORAGE,
    VORNetLSTM_PINN,
    VORPhysicsSimulator,
)

_SEQ_LEN = 900          # ~30 s at the 30 FPS deployment rate (paper: seq length = 900).
_TRAIN_VAL_SPLIT = 0.85


# =============================================================================
# 损失函数 (Eq. 30: L = L_data + lambda_s * L_state)
# =============================================================================

def state_supervision_loss(c_hat, s_hat, c_sim, s_sim):
    """L_state (Eq. 32): MSE between predicted and simulator (canal, storage) states."""
    c_scale = (CANAL_GAIN_BASE * 100.0) ** 2 + 1e-6
    s_scale = (CANAL_GAIN_BASE * 20.0) ** 2 + 1e-6
    return (F.mse_loss(c_hat, c_sim) / c_scale
            + F.mse_loss(s_hat, s_sim) / s_scale)


# =============================================================================
# 合成数据生成 (精简场景)
# =============================================================================

def _make_dynamic_trajectory(scene: int, t: np.ndarray, rng) -> np.ndarray:
    """生成动态轨迹 (正弦、阶跃-停止、快慢组合)"""
    if scene == 0:          # 正弦
        freq = rng.uniform(0.3, 2.0)
        amp = np.clip(rng.normal(50.0, 40.0), 10.0, 200.0)
        ph = rng.uniform(0, 2 * np.pi, 3)
        omega = amp * np.column_stack([
            np.sin(2 * np.pi * freq * t + ph[0]),
            np.sin(2 * np.pi * freq * 0.7 * t + ph[1]) * 0.5,
            np.sin(2 * np.pi * freq * 1.3 * t + ph[2]) * 0.3,
        ])
        return omega
    elif scene == 1:        # 阶跃-停止 (post-rotatory decay)
        omega = np.zeros((_SEQ_LEN, 3))
        t0 = int(_SEQ_LEN * rng.uniform(0.1, 0.3))
        t1 = int(_SEQ_LEN * rng.uniform(0.5, 0.9))
        omega[t0:t1, 0] = np.clip(rng.normal(0.0, 70.0), -150.0, 150.0)
        omega[t0:t1, 1] = np.clip(rng.normal(0.0, 35.0), -80.0, 80.0)
        omega[t0:t1, 2] = np.clip(rng.normal(0.0, 18.0), -40.0, 40.0)
        return omega
    else:                   # scene==2: 慢-快组合
        mid = _SEQ_LEN // 2
        omega = np.zeros((_SEQ_LEN, 3))
        omega[:mid, 0] = 30 * np.sin(2 * np.pi * 0.3 * t[:mid])
        omega[mid:, 0] = 250 * np.sin(2 * np.pi * 1.5 * t[mid:])
        omega[:, 1] = 20 * np.sin(2 * np.pi * 0.5 * t)
        return omega


def generate_sequences(n_dynamic: int = 400, dt: float = DT):
    """生成训练数据：动态 + 常速 + 阶跃-停止"""
    sim = VORPhysicsSimulator(dt=dt)
    rng = np.random.default_rng(seed=42)
    t = np.linspace(0, _SEQ_LEN * dt, _SEQ_LEN)

    omega_list, comp_list, state_list = [], [], []

    def add(omega_seq: np.ndarray) -> None:
        states, outputs = sim.simulate(omega_seq)
        omega_list.append(omega_seq.astype(np.float32))
        comp_list.append(outputs)
        state_list.append(states)

    # 动态场景 (三种)
    for i in range(n_dynamic):
        add(_make_dynamic_trajectory(i % 3, t, rng))

    # 常速段 (各轴, 多种速度)
    for speed in (10., 50., 100., 200., 300., 400.):
        for axis in range(3):
            for sign in (+1, -1):
                omega = np.zeros((_SEQ_LEN, 3))
                omega[:, axis] = sign * speed
                add(omega)

    # 额外 yaw 常速 (覆盖常用范围)
    for speed in (60., 80., 100., 150.):
        for sign in (+1, -1):
            omega = np.zeros((_SEQ_LEN, 3))
            omega[:, 0] = sign * speed
            add(omega)

    # 阶跃-停止段 (post-rotatory decay, paper motion type iv)
    for speed in (30., 60., 100., 150., 200.):
        for axis in range(3):
            for sign in (+1, -1):
                for stop_frac in (0.27, 0.40, 0.55):
                    stop_step = int(_SEQ_LEN * stop_frac)
                    omega = np.zeros((_SEQ_LEN, 3))
                    omega[:stop_step, axis] = sign * speed
                    add(omega)

    return omega_list, comp_list, state_list


def compute_omega_dot(omega: np.ndarray, dt: float = DT) -> np.ndarray:
    omega = np.asarray(omega, dtype=np.float32)
    omega_dot = np.zeros_like(omega)
    omega_dot[1:] = (omega[1:] - omega[:-1]) / dt
    return omega_dot


# =============================================================================
# 推理辅助函数
# =============================================================================

def _run_seq(model, omega_raw, feat_mean, feat_std, tgt_std, device,
             chunk_size: int = 200,
             feat_mean_dot=None, feat_std_dot=None) -> np.ndarray:
    T = len(omega_raw)
    omega_n = (omega_raw.astype(np.float32) - feat_mean) / feat_std
    omega_dot = compute_omega_dot(omega_raw)
    omega_dot_n = (omega_dot - feat_mean_dot) / feat_std_dot
    inp = np.concatenate([omega_n, omega_dot_n], axis=-1).astype(np.float32)

    x = torch.from_numpy(inp).unsqueeze(0).to(device)
    preds, h = [], None
    model.eval()
    with torch.no_grad():
        for s in range(0, T, chunk_size):
            comp, _, _, h = model(x[:, s:s + chunk_size], h)
            preds.append(comp)
    return torch.cat(preds, dim=1).squeeze(0).cpu().numpy()


# =============================================================================
# 训练流程
# =============================================================================

def _to_serialisable(obj):
    if isinstance(obj, dict):
        return {k: _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serialisable(x) for x in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def train_vor_lstm(save_path: str = DEFAULT_MODEL_PATH,
                   n_dynamic: int = 400,
                   n_epochs: int = 800,
                   batch_size: int = 4,
                   lr: float = 5e-4,
                   ode_weight: float = 0.4,   # lambda_s in Eq. (30)
                   plot_dir: str = DEFAULT_PLOT_DIR,
                   verbose: bool = True) -> dict:
    os.makedirs(plot_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if verbose:
        print(f'\n精简训练 | device={device} epochs={n_epochs} batch={batch_size} lr={lr}')
        print(f'  lambda_s (ode_weight)={ode_weight}')

    # 生成数据
    t0 = time.time()
    omega_list, comp_list, state_list = generate_sequences(n_dynamic=n_dynamic)
    n_traj = len(omega_list)
    if verbose:
        print(f'数据生成: {n_traj} 条轨迹, 耗时 {time.time()-t0:.1f}s')

    # 归一化参数
    all_omega = np.vstack(omega_list)
    feat_mean = all_omega.mean(axis=0).astype(np.float32)
    feat_std = (all_omega.std(axis=0) + 1e-8).astype(np.float32)

    omega_dot_list = [compute_omega_dot(om) for om in omega_list]
    all_omega_dot = np.vstack(omega_dot_list)
    feat_mean_dot = all_omega_dot.mean(axis=0).astype(np.float32)
    feat_std_dot = (all_omega_dot.std(axis=0) + 1e-8).astype(np.float32)

    tgt_std = np.array([MAX_X_PX, MAX_Y_PX], dtype=np.float32)

    # 堆叠数据
    n = len(omega_list)
    X_all = np.zeros((n, _SEQ_LEN, 6), dtype=np.float32)
    y_all = np.zeros((n, _SEQ_LEN, 2), dtype=np.float32)
    St_all = np.zeros((n, _SEQ_LEN, 6), dtype=np.float32)
    for i, (omega, comp, states) in enumerate(zip(omega_list, comp_list, state_list)):
        omega_n = (omega - feat_mean) / feat_std
        omega_dot_n = (compute_omega_dot(omega) - feat_mean_dot) / feat_std_dot
        X_all[i] = np.concatenate([omega_n, omega_dot_n], axis=-1)
        y_all[i] = comp / tgt_std
        St_all[i] = states

    # 训练/验证分割
    rng = np.random.default_rng(seed=0)
    perm = rng.permutation(n_traj)
    n_tr = int(_TRAIN_VAL_SPLIT * n_traj)
    tr_idx, va_idx = perm[:n_tr], perm[n_tr:]

    X_tr = torch.from_numpy(X_all[tr_idx]).to(device)
    y_tr = torch.from_numpy(y_all[tr_idx]).to(device)
    St_tr = torch.from_numpy(St_all[tr_idx]).to(device)
    X_va = torch.from_numpy(X_all[va_idx]).to(device)
    y_va = torch.from_numpy(y_all[va_idx]).to(device)
    tgt_std_t = torch.from_numpy(tgt_std).to(device)

    # 轨迹权重 (偏向低速)
    peaks_tr = np.array([np.max(np.abs(omega_list[i])) for i in tr_idx], dtype=np.float32)
    w_data_tr_np = np.clip(80.0 / np.maximum(peaks_tr, 20.0), 0.4, 4.0)
    W_data_tr = torch.from_numpy(w_data_tr_np).to(device)

    model = VORNetLSTM_PINN(hidden=64, num_layers=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=300, T_mult=2, eta_min=lr*0.05)
    criterion = nn.MSELoss()

    history = {'epoch': [], 'loss_data': [], 'loss_state': [], 'val_loss': []}
    best_val = float('inf')
    best_state = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        perm_epoch = torch.randperm(n_tr, device=X_tr.device)
        sums = {'data': 0.0, 'state': 0.0}
        n_batches = 0
        for b in range(0, n_tr, batch_size):
            idx = perm_epoch[b:b+batch_size]
            bX = X_tr[idx]
            by = y_tr[idx]
            bSt = St_tr[idx]
            bW = W_data_tr[idx]

            optimizer.zero_grad(set_to_none=True)
            comp, c_hat, s_hat, _ = model(bX, None)

            err_sq = ((comp / tgt_std_t - by) ** 2).mean(dim=(1,2))
            l_data = (err_sq * bW).sum() / (bW.sum() + 1e-8)
            l_state = state_supervision_loss(c_hat, s_hat, bSt[...,:3], bSt[...,3:])

            # Eq. (30): L = L_data + lambda_s * L_state
            loss = l_data + ode_weight * l_state
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            sums['data'] += l_data.item()
            sums['state'] += l_state.item()
            n_batches += 1

        n_batches = max(1, n_batches)
        ep_losses = {k: v/n_batches for k,v in sums.items()}

        # 验证
        model.eval()
        with torch.no_grad():
            val_comp, _, _, _ = model(X_va, None)
            val_loss = criterion(val_comp / tgt_std_t, y_va).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}

        scheduler.step()
        history['epoch'].append(epoch)
        history['loss_data'].append(ep_losses['data'])
        history['loss_state'].append(ep_losses['state'])
        history['val_loss'].append(val_loss)

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"  {epoch:5d} L_data={ep_losses['data']:.5f} "
                  f"L_state={ep_losses['state']:.5f} val={val_loss:.5f}")

    # 保存最佳模型
    if best_state is not None:
        model.load_state_dict(best_state)
    checkpoint = {
        'model_state': model.state_dict(),
        'feat_mean': feat_mean,
        'feat_std': feat_std,
        'feat_mean_dot': feat_mean_dot,
        'feat_std_dot': feat_std_dot,
        'tgt_std': tgt_std,
        'config': {'hidden': 64, 'num_layers': 1, 'best_val': float(best_val)},
    }
    torch.save(checkpoint, save_path)
    if verbose:
        print(f'模型保存: {save_path}')

    # 保存历史 & 绘图
    hist_path = save_path.replace('.pth', '_history.json')
    with open(hist_path, 'w') as f:
        json.dump(_to_serialisable(history), f, indent=2)
    _plot_training(history, plot_dir, verbose)
    return history


def _plot_training(history: dict, plot_dir: str, verbose: bool) -> None:
    if not _MATPLOTLIB_AVAILABLE:
        return
    epochs = history['epoch']
    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    fig.suptitle('Training Losses', fontweight='bold')
    axes[0].semilogy(epochs, history['loss_data'], label='L_data', color='#378ADD')
    axes[0].semilogy(epochs, history['val_loss'], label='Validation', color='#E24B4A', ls='--')
    axes[0].set_title('Data + Validation')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].semilogy(epochs, history['loss_state'], label='L_state (ODE)', color='#1D9E75')
    axes[1].set_title('Physics loss')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(plot_dir, 'training_curves.png')
    plt.savefig(out, dpi=150)
    plt.close()
    if verbose:
        print(f'训练曲线保存: {out}')


# =============================================================================
# 验证 (只保留阶跃响应 + 正弦追踪 + 衰减图)
# =============================================================================

def run_validation(model_path: str = DEFAULT_MODEL_PATH,
                   plot_dir: str = DEFAULT_PLOT_DIR,
                   verbose: bool = True) -> dict:
    os.makedirs(plot_dir, exist_ok=True)
    device = torch.device('cpu')
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    model = VORNetLSTM_PINN(hidden=ckpt['config'].get('hidden',64),
                            num_layers=ckpt['config'].get('num_layers',1))
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    sim = VORPhysicsSimulator(dt=DT)
    feat_mean = ckpt['feat_mean']
    feat_std = ckpt['feat_std']
    feat_mean_dot = ckpt['feat_mean_dot']
    feat_std_dot = ckpt['feat_std_dot']
    tgt_std = ckpt['tgt_std']
    chunk_size = 200

    def predict(omega_raw):
        return _run_seq(model, omega_raw, feat_mean, feat_std, tgt_std,
                        device, chunk_size,
                        feat_mean_dot, feat_std_dot)

    metrics = {}

    # 阶跃响应 RMSE
    step_results = []
    for spd in (30., 60., 100., 150.):
        omega = np.zeros((_SEQ_LEN,3), dtype=np.float32)
        omega[:,0] = spd
        _, ref = sim.simulate(omega)
        pred = predict(omega)
        rmse_x = float(np.sqrt(np.mean((pred[:,0]-ref[:,0])**2)))
        step_results.append({'speed':spd, 'rmse_x':rmse_x})
    metrics['step_rmse'] = step_results

    # 正弦追踪 (1Hz ±60 dps)
    t = np.arange(_SEQ_LEN)*DT
    omega_sin = np.zeros((_SEQ_LEN,3))
    omega_sin[:,0] = 60.0 * np.sin(2*np.pi*1.0*t)
    _, ref_sin = sim.simulate(omega_sin)
    pred_sin = predict(omega_sin)
    ref_steady = ref_sin[1500:,0]
    pred_steady = pred_sin[1500:,0]
    rmse_sin = float(np.sqrt(np.mean((pred_steady - ref_steady)**2)))
    xc = np.correlate(pred_steady - pred_steady.mean(),
                      ref_steady - ref_steady.mean(), mode='full')
    lag_steps = int(np.argmax(xc) - (len(ref_steady)-1))
    lag_ms = int(lag_steps * DT * 1000)
    metrics['sin_tracking'] = {'rmse_px': round(rmse_sin,3), 'phase_lag_ms': lag_ms}

    # 绘图: 正弦 + 衰减图
    _plot_validation(model, feat_mean, feat_std, tgt_std, plot_dir, verbose,
                     chunk_size, feat_mean_dot, feat_std_dot)

    # 保存指标
    metrics_path = os.path.join(plot_dir, 'validation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(_to_serialisable(metrics), f, indent=2)
    if verbose:
        print(f'验证指标保存: {metrics_path}')
    return metrics


def _plot_validation(model, feat_mean, feat_std, tgt_std, plot_dir, verbose,
                     chunk_size, feat_mean_dot, feat_std_dot) -> None:
    if not _MATPLOTLIB_AVAILABLE:
        return
    sim = VORPhysicsSimulator(dt=DT)
    t = np.arange(_SEQ_LEN) * DT
    device = next(model.parameters()).device

    def predict(omega_raw):
        return _run_seq(model, omega_raw, feat_mean, feat_std, tgt_std,
                        device, chunk_size, feat_mean_dot, feat_std_dot)

    fig, axes = plt.subplots(1, 2, figsize=(14,5))
    fig.suptitle('Validation: Sine Tracking & Decay', fontweight='bold')

    # 正弦追踪
    omega_sin = np.zeros((_SEQ_LEN,3))
    omega_sin[:,0] = 60.0 * np.sin(2*np.pi*1.0*t)
    _, ref_sin = sim.simulate(omega_sin)
    pred_sin = predict(omega_sin)
    sl = slice(2000, _SEQ_LEN)
    axes[0].plot(t[sl], ref_sin[sl,0], '--', color='#888780', label='ODE')
    axes[0].plot(t[sl], pred_sin[sl,0], '-', color='#378ADD', label='LSTM')
    rmse = np.sqrt(np.mean((pred_sin[2000:,0]-ref_sin[2000:,0])**2))
    axes[0].set_title(f'1 Hz ±60 dps  RMSE={rmse:.3f}px')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Compensation (px)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 短暂头转+衰减 (100 dps × 2s)
    omega_decay = np.zeros((_SEQ_LEN,3))
    stop_step = int(2.0/DT)
    omega_decay[:stop_step,0] = 100.0
    _, ref_decay = sim.simulate(omega_decay)
    pred_decay = predict(omega_decay)
    axes[1].plot(t, ref_decay[:,0], '--', color='#888780', label='ODE')
    axes[1].plot(t, pred_decay[:,0], '-', color='#1D9E75', label='LSTM')
    axes[1].axvline(2.0, color='#E24B4A', ls=':', label='Motion stops at 2s')
    axes[1].set_title('100 dps → stop at 2s')
    axes[1].set_xlabel('Time (s)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(plot_dir, 'validation_plots.png')
    plt.savefig(out, dpi=150)
    plt.close()
    if verbose:
        print(f'验证图保存: {out}')


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='精简 LSTM-PINN VOR')
    parser.add_argument('--mode', choices=('train','validate','all'), default='all')
    parser.add_argument('--save', default=DEFAULT_MODEL_PATH)
    parser.add_argument('--load', default=None)
    parser.add_argument('--plot_dir', default=DEFAULT_PLOT_DIR)
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--retrain', action='store_true')
    args = parser.parse_args()

    model_path = args.load or args.save
    if args.mode == 'train':
        train_vor_lstm(save_path=model_path, n_epochs=args.epochs, plot_dir=args.plot_dir)
    elif args.mode == 'validate':
        run_validation(model_path=model_path, plot_dir=args.plot_dir)
    else:
        if args.retrain or not os.path.exists(model_path):
            train_vor_lstm(save_path=model_path, n_epochs=args.epochs, plot_dir=args.plot_dir)
        run_validation(model_path=model_path, plot_dir=args.plot_dir)


if __name__ == '__main__':
    main()