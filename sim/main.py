"""
シミュレーション実行・可視化

CFG の "mode" を切り替えて実験する:
    "cascade"    : カスケード PID で倒立振り子制御
    "inner_only" : 内側 PI ループ単独テスト（定数 omega_ref への追従確認）

実行方法: python sim/main.py   (swing/ ディレクトリから)
"""

import sys
from pathlib import Path

_here = Path(__file__).parent
sys.path.insert(0, str(_here.parent))  # swing/     → params
sys.path.insert(0, str(_here))         # swing/sim/ → plant, motor, ...

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim_mod
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle

from params import RobotParams, ControllerParams
from plant import rk4_step
from motor import pwm_to_torque
from sensors import IMU, Encoder
from controller import OuterLoop, InnerLoop

# ============================================================
# 実験設定  ← ここを書き換えて実験する
# ============================================================
CFG = {
    "mode":            "cascade",  # "cascade" | "inner_only"
    "duration":        5.0,        # シミュレーション時間 [s]
    "theta_init":      0.05,       # 初期傾斜角 [rad]（cascade の外乱）
    "omega_ref_const": 10.0,       # inner_only: 目標タイヤ角速度 [rad/s]
    "imu_noise_std":   0.001,      # IMU ガウスノイズ標準偏差 [rad]
    "enc_noise_std":   0.01,       # エンコーダー ノイズ標準偏差 [rad]
}

rp = RobotParams()
cp = ControllerParams()


# ============================================================
# シミュレーション
# ============================================================
def run_simulation(cfg: dict, rp: RobotParams, cp: ControllerParams) -> np.ndarray:
    """
    Returns
    -------
    log : ndarray, shape (N, 7)
        columns: t, theta, theta_dot, phi, phi_dot, omega_ref, pwm
    """
    theta_init = cfg["theta_init"] if cfg["mode"] == "cascade" else 0.0
    state = np.array([theta_init, 0.0, 0.0, 0.0], dtype=float)

    imu   = IMU(rp,     noise_std=cfg["imu_noise_std"])
    enc   = Encoder(rp, noise_std=cfg["enc_noise_std"])
    outer = OuterLoop(cp)
    inner = InnerLoop(cp)

    steps = int(cfg["duration"] / rp.sim_dt)
    log   = np.empty((steps, 7))
    t     = 0.0
    pwm   = 0.0

    for i in range(steps):
        theta_true, _, phi_true, phi_dot_true = state

        imu_fired = imu.update(t, theta_true)
        enc_fired = enc.update(t, phi_true)

        if cfg["mode"] == "cascade":
            if imu_fired:
                alpha_ref = outer.step(imu.theta, rp.imu_dt)
                inner.update_omega_ref(alpha_ref, rp.imu_dt)
            if enc_fired:
                pwm = inner.step(enc.omega, rp.encoder_dt)
        else:  # inner_only
            inner.omega_ref = cfg["omega_ref_const"]
            if enc_fired:
                pwm = inner.step(enc.omega, rp.encoder_dt)

        tau   = pwm_to_torque(pwm, phi_dot_true, rp)
        state = rk4_step(state, tau, rp)
        t    += rp.sim_dt

        log[i] = [t, state[0], state[1], state[2], state[3], inner.omega_ref, pwm]

        # 重心が地面にめり込んだら停止
        cy = rp.r + rp.L * np.cos(state[0])
        if cy <= 0.0:
            print(f"[sim] 重心が地面に達したため t={t:.3f}s で停止")
            return log[:i + 1]

    return log


# ============================================================
# 可視化
# ============================================================
def visualize(log: np.ndarray, rp: RobotParams, cfg: dict):
    t_arr         = log[:, 0]
    theta_arr     = log[:, 1]
    phi_arr       = log[:, 3]
    phi_dot_arr   = log[:, 4]
    omega_ref_arr = log[:, 5]
    pwm_arr       = log[:, 6]

    x_wheel_arr = phi_arr * rp.r  # no-slip での車輪 x 位置 [m]

    # ---- レイアウト ----
    fig = plt.figure(figsize=(15, 7))
    gs  = GridSpec(2, 2, figure=fig, width_ratios=[1.4, 1], hspace=0.5, wspace=0.45)
    ax_anim  = fig.add_subplot(gs[:, 0])
    ax_theta = fig.add_subplot(gs[0, 1])
    ax_omega = fig.add_subplot(gs[1, 1])
    fig.suptitle(f"simulator  [mode: {cfg['mode']}]", fontsize=13)

    # ---- 右上: θ（左軸・青）と ω_ref（右軸・橙） ----
    ax_theta.set_title("θ vs ω_ref", fontsize=10)
    ax_theta.set_xlabel("t [s]")
    ax_theta.set_ylabel("θ [deg]", color="tab:blue")
    ax_theta.plot(t_arr, np.degrees(theta_arr), color="tab:blue")
    ax_theta.tick_params(axis="y", labelcolor="tab:blue")
    ax_theta.grid(True, alpha=0.3)
    ax_t2 = ax_theta.twinx()
    ax_t2.set_ylabel("ω_ref [rad/s]", color="tab:orange")
    ax_t2.plot(t_arr, omega_ref_arr, color="tab:orange", linestyle="--")
    ax_t2.tick_params(axis="y", labelcolor="tab:orange")

    # ---- 右下: ω_ref vs ω_actual ----
    ax_omega.set_title("target vs actual", fontsize=10)
    ax_omega.set_xlabel("t [s]")
    ax_omega.set_ylabel("[rad/s]")
    ax_omega.plot(t_arr, omega_ref_arr, color="tab:orange", linestyle="--", label="ω_ref")
    ax_omega.plot(t_arr, phi_dot_arr,   color="tab:green",                  label="ω_actual")
    ax_omega.legend(fontsize=9)
    ax_omega.grid(True, alpha=0.3)

    # ---- アニメーション ----
    r, L = rp.r, rp.L
    stride    = max(1, len(log) // 500)  # 約 500 フレームに間引く
    frame_idx = np.arange(0, len(log), stride)

    ax_anim.set_aspect("equal")
    ax_anim.set_title("animation", fontsize=10)
    ax_anim.set_xlabel("x [m]")
    ax_anim.set_ylabel("y [m]")
    ax_anim.set_ylim(-0.03, L * 2.5 + r)

    # 地面（広い範囲で静的に描画、xlim 変化に追従する）
    ax_anim.axhline(0.0, color="black", linewidth=2, zorder=3)
    ax_anim.fill_between([-200, 200], -0.03, 0.0, color="#c8a96e", alpha=0.5, zorder=2)

    # 動的要素
    wheel   = Circle((0, r), r, fill=False, color="tab:blue", linewidth=2.5, zorder=5)
    ax_anim.add_patch(wheel)
    rod,     = ax_anim.plot([], [], "-", color="tab:red",  linewidth=3,   zorder=6)
    com_dot, = ax_anim.plot([], [], "o", color="tab:red",  markersize=10, zorder=7)
    axle_dot,= ax_anim.plot([], [], "o", color="gray",     markersize=5,  zorder=7)
    pwm_lbl  = ax_anim.text(0, 0, "", fontsize=9, ha="center", va="top", zorder=8,
                             bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                       ec="gray", alpha=0.85))
    time_lbl = ax_anim.text(0.02, 0.97, "", transform=ax_anim.transAxes,
                             fontsize=9, va="top")

    def _animate(i):
        idx   = frame_idx[i]
        theta = theta_arr[idx]
        pwm   = pwm_arr[idx]
        t_cur = t_arr[idx]
        xw    = x_wheel_arr[idx]

        # 重心位置（車軸中心 + 傾斜方向 L）
        cx = xw + L * np.sin(theta)
        cy = r  + L * np.cos(theta)

        ax_anim.set_xlim(xw - 0.4, xw + 0.4)
        wheel.center    = (xw, r)
        rod.set_data([xw, cx], [r, cy])
        com_dot.set_data([cx], [cy])
        axle_dot.set_data([xw], [r])
        pwm_lbl.set_position((xw, r - r * 0.65))
        pwm_lbl.set_text(f"PWM: {pwm:+.0f}")
        time_lbl.set_text(f"t = {t_cur:.2f} s")

        return wheel, rod, com_dot, axle_dot, pwm_lbl, time_lbl

    interval_ms = max(16, int(rp.sim_dt * stride * 1000))
    _anim = anim_mod.FuncAnimation(          # 変数に束縛しないと GC される
        fig, _animate, frames=len(frame_idx),
        interval=interval_ms, blit=False, repeat=True,
    )

    plt.show()


# ============================================================
# エントリーポイント
# ============================================================
if __name__ == "__main__":
    print(f"[sim] mode={CFG['mode']}  duration={CFG['duration']}s")
    log = run_simulation(CFG, rp, cp)
    print(f"[sim] done  steps={len(log)}")
    visualize(log, rp, CFG)
