"""
センサーモデル: IMU・エンコーダー

各センサーは設定した制御周期（dt）でのみ値を更新するゼロ次ホールド（ZOH）。
ガウスノイズはオプション（noise_std=0.0 でノイズなし）。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from params import RobotParams


class IMU:
    """
    傾斜角センサー。
    update() を毎物理ステップ呼ぶと、imu_dt ごとに theta を更新する。
    """

    def __init__(self, p: RobotParams, noise_std: float = 0.0):
        self._dt = p.imu_dt
        self._noise_std = noise_std
        self._theta: float = 0.0
        self._last_t: float = -p.imu_dt   # 初回即更新のため

    def update(self, t: float, theta_true: float) -> bool:
        """
        物理時刻 t と真値 theta_true を渡す。
        サンプリングタイミングなら True を返す（新しい値を取得した）。
        """
        if t >= self._last_t + self._dt:
            noise = np.random.normal(0.0, self._noise_std) if self._noise_std > 0.0 else 0.0
            self._theta = theta_true + noise
            self._last_t = t
            return True
        return False

    @property
    def theta(self) -> float:
        """最新のサンプル値 [rad]"""
        return self._theta


class Encoder:
    """
    タイヤ磁気エンコーダー。
    encoder_dt ごとに phi をサンプリングし、1階差分で角速度を算出する。
    """

    def __init__(self, p: RobotParams, noise_std: float = 0.0):
        self._dt = p.encoder_dt
        self._noise_std = noise_std
        self._omega: float = 0.0
        self._phi_prev: float = 0.0
        self._last_t: float = -p.encoder_dt

    def update(self, t: float, phi_true: float) -> bool:
        """
        物理時刻 t と真値 phi_true を渡す。
        サンプリングタイミングなら True を返す（omega が更新された）。
        """
        if t >= self._last_t + self._dt:
            noise = np.random.normal(0.0, self._noise_std) if self._noise_std > 0.0 else 0.0
            phi_measured = phi_true + noise
            self._omega = (phi_measured - self._phi_prev) / self._dt
            self._phi_prev = phi_measured
            self._last_t = t
            return True
        return False

    @property
    def omega(self) -> float:
        """最新の角速度サンプル [rad/s]"""
        return self._omega
