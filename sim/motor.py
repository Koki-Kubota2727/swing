"""
モーターモデル: PWM → 電圧 → トルク

DCモーターの標準モデル:
  V_eff = duty * V_supply - K_e * phi_dot   (逆起電力を差し引いた有効電圧)
  I     = V_eff / R_motor                   (電機子電流)
  tau   = K_t * I                           (発生トルク)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from params import RobotParams


def pwm_to_torque(pwm: float, phi_dot: float, p: RobotParams) -> float:
    """
    PWM値とタイヤ角速度からモータートルクを計算する。

    Args:
        pwm     : PWM指令値（-pwm_max 〜 +pwm_max）
        phi_dot : タイヤ角速度 [rad/s]
        p       : ロボットパラメータ

    Returns:
        tau: モータートルク [N·m]
    """
    pwm_clipped = np.clip(pwm, -p.pwm_max, p.pwm_max)
    duty = pwm_clipped / p.pwm_max                        # -1 〜 +1
    voltage = duty * p.V_supply                           # 印加電圧 [V]
    voltage_eff = voltage - p.K_e * phi_dot               # 逆起電力を除いた有効電圧
    current = voltage_eff / p.R_motor                     # 電機子電流 [A]
    tau = p.K_t * current                                 # トルク [N·m]
    return tau
