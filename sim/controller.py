"""
カスケードPIDコントローラー

外側ループ: θ → α_ref [rad/s²]  （PID）
外側→内側: ω_ref += α_ref × dt
内側ループ: Δω → PWM             （PI）

外側と内側の周期が異なる場合は main.py 側で個別に呼ぶ:
    if imu.update(...):
        alpha_ref = outer.step(theta, p.imu_dt)
        inner.update_omega_ref(alpha_ref, p.imu_dt)
    if encoder.update(...):
        pwm = inner.step(encoder.omega, p.encoder_dt)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from params import ControllerParams


class OuterLoop:
    """傾斜角 PID: Δθ → α_ref（タイヤ目標角加速度）"""

    def __init__(self, p: ControllerParams):
        self.Kp = p.outer_Kp
        self.Ki = p.outer_Ki
        self.Kd = p.outer_Kd
        self.alpha_max = p.alpha_max
        self._integral: float = 0.0
        self._prev_error: float = 0.0

    def step(self, theta: float, dt: float) -> float:
        """
        Args:
            theta: 機体傾斜角 [rad]（θ_ref = 0）
            dt   : 外側ループ更新周期 [s]
        Returns:
            alpha_ref: タイヤ目標角加速度 [rad/s²]
        """
        error = theta  # θ_ref = 0　//ここの正負調節
        self._integral += error * dt
        d_error = (error - self._prev_error) / dt
        self._prev_error = error

        alpha_ref = self.Kp * error + self.Ki * self._integral + self.Kd * d_error
        alpha_ref = float(np.clip(alpha_ref, -self.alpha_max, self.alpha_max))

        # 飽和時は積分を戻す（anti-windup）
        if abs(alpha_ref) >= self.alpha_max:
            self._integral -= error * dt

        return alpha_ref

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0


class InnerLoop:
    """タイヤ角速度 PI: Δω → PWM"""

    def __init__(self, p: ControllerParams):
        self.Kp = p.inner_Kp
        self.Ki = p.inner_Ki
        self.pwm_max = p.pwm_max
        self.omega_ref: float = 0.0
        self._integral: float = 0.0

    def update_omega_ref(self, alpha_ref: float, dt: float) -> None:
        """α_ref を積分して ω_ref を更新する（外側ループ周期で呼ぶ）"""
        self.omega_ref += alpha_ref * dt

    def step(self, omega_meas: float, dt: float) -> float:
        """
        Args:
            omega_meas: 計測タイヤ角速度 [rad/s]
            dt        : 内側ループ更新周期 [s]
        Returns:
            pwm: PWM指令値（-pwm_max 〜 +pwm_max）
        """
        error = self.omega_ref - omega_meas
        self._integral += error * dt

        pwm = self.Kp * error + self.Ki * self._integral
        pwm = float(np.clip(pwm, -self.pwm_max, self.pwm_max))

        # 飽和時は積分を戻す（anti-windup）
        if abs(pwm) >= self.pwm_max:
            self._integral -= error * dt

        return pwm

    def reset(self) -> None:
        self.omega_ref = 0.0
        self._integral = 0.0


def cascade_step(
    theta: float,
    omega_wheel: float,
    outer: OuterLoop,
    inner: InnerLoop,
    dt: float,
) -> float:
    """
    外側・内側ループを同一周期で1ステップ更新する。

    Returns:
        pwm: PWM指令値
    """
    alpha_ref = outer.step(theta, dt)
    inner.update_omega_ref(alpha_ref, dt)
    return inner.step(omega_wheel, dt)
