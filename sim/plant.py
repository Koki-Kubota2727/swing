"""
物理モデル（倒立振り子の運動方程式 + RK4積分）

状態ベクトル: [theta, theta_dot, phi, phi_dot]
  theta     : 機体傾斜角 [rad]  （直立=0、前傾が正）
  theta_dot : 機体角速度 [rad/s]
  phi       : タイヤ回転角 [rad]
  phi_dot   : タイヤ角速度 [rad/s]

ラグランジアン法による運動方程式（no-slip前提）:
  [M11  M12] [theta_ddot]   [M_b*g*L*sin(theta) - tau        ]
  [M12  M22] [phi_ddot  ] = [tau - F_friction*r               ]

スリップ判定:
  no-slip に必要な地面摩擦力が mu_s*N を超えたら動摩擦に切り替え
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from params import RobotParams


def _mass_matrix(theta: float, p: RobotParams):
    M11 = p.I_b + p.M_b * p.L ** 2
    M12 = p.M_b * p.L * p.r * np.cos(theta)
    M22 = p.I_w + p.M_total * p.r ** 2
    det = M11 * M22 - M12 ** 2
    return M11, M12, M22, det


def _solve_no_slip(theta: float, theta_dot: float, phi_dot: float,
                   tau: float, p: RobotParams):
    """
    スリップなしとして加速度を解く。
    返り値: (theta_ddot, phi_ddot, f_friction)
      f_friction: no-slip を維持するために必要な地面摩擦力 [N]
    """
    M11, M12, M22, det = _mass_matrix(theta, p)

    centrifugal = p.M_b * p.L * theta_dot ** 2 * np.sin(theta)

    f1 = p.M_b * p.g * p.L * np.sin(theta) - tau          # θ 方程式の右辺
    f2 = tau + M12 / p.r * centrifugal / p.M_total        # φ 方程式（近似）

    # より正確な形：Coriolis項を含む
    # f2 を素直に tau のみにして centrifugal を分離する形
    f1 = p.M_b * p.g * p.L * np.sin(theta) - tau
    f2 = tau

    theta_ddot = (M22 * f1 - M12 * f2) / det
    phi_ddot = (M11 * f2 - M12 * f1) / det

    # no-slip を維持するために必要な地面摩擦力
    # タイヤの運動方程式: I_w*phi_ddot = tau - f_friction*r
    f_friction = (tau - p.I_w * phi_ddot) / p.r

    return theta_ddot, phi_ddot, f_friction


def _solve_slip(theta: float, tau: float, phi_dot: float, p: RobotParams):
    """
    スリップ中の加速度を解く。
    地面摩擦力 = mu_k * N * sign(phi_dot)
    """
    N = p.M_total * p.g
    f_friction = p.mu_k * N * np.sign(phi_dot) if abs(phi_dot) > 1e-6 else 0.0

    # タイヤ（地面摩擦力がかかる）
    phi_ddot = (tau - f_friction * p.r) / p.I_w

    # 機体（近似：スリップ中はタイヤとの連成を無視）
    M11 = p.I_b + p.M_b * p.L ** 2
    theta_ddot = (p.M_b * p.g * p.L * np.sin(theta) - tau) / M11

    return theta_ddot, phi_ddot


def derivatives(state: np.ndarray, tau: float, p: RobotParams) -> np.ndarray:
    """状態微分を返す。スリップ判定込み。"""
    theta, theta_dot, phi, phi_dot = state

    theta_ddot, phi_ddot, f_friction = _solve_no_slip(
        theta, theta_dot, phi_dot, tau, p
    )

    # スリップ判定
    N = p.M_total * p.g
    if abs(f_friction) > p.mu_s * N:
        theta_ddot, phi_ddot = _solve_slip(theta, tau, phi_dot, p)

    return np.array([theta_dot, theta_ddot, phi_dot, phi_ddot])


def rk4_step(state: np.ndarray, tau: float, p: RobotParams) -> np.ndarray:
    """RK4 で 1ステップ（p.sim_dt）進める。"""
    dt = p.sim_dt
    k1 = derivatives(state, tau, p)
    k2 = derivatives(state + 0.5 * dt * k1, tau, p)
    k3 = derivatives(state + 0.5 * dt * k2, tau, p)
    k4 = derivatives(state + dt * k3, tau, p)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
