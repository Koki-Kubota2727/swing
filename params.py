from dataclasses import dataclass


@dataclass
class RobotParams:
    # --- 機体 ---
    M_b: float = 0.5        # 機体質量 [kg]
    L: float = 0.10         # 車軸中心→重心 距離 [m]
    I_b: float = 0.005      # 重心周りの慣性モーメント [kg·m²]

    # --- タイヤ ---
    M_w: float = 0.05       # タイヤ質量 [kg]
    r: float = 0.05         # タイヤ半径 [m]
    I_w: float = 0.0001     # タイヤ慣性モーメント [kg·m²]（小→入力切ると即停止）

    # --- タイヤ-地面 摩擦 ---
    mu_s: float = 0.6   # 静止摩擦係数
    mu_k: float = 0.4       # 動摩擦係数

    # --- モーター ---
    K_t: float = 0.05       # トルク定数 [N·m/A]
    K_e: float = 0.05       # 逆起電力定数 [V·s/rad]
    R_motor: float = 2.0    # モーター内部抵抗 [Ω]
    V_supply: float = 12.0  # 電源電圧 [V]
    pwm_max: float = 255.0  # PWM最大値

    # --- センサー周期 ---
    imu_dt: float = 0.010      # IMU サンプリング周期 [s]
    encoder_dt: float = 0.010  # エンコーダー サンプリング周期 [s]

    # --- シミュレーション ---
    sim_dt: float = 0.001   # 物理積分ステップ [s]
    g: float = 9.81         # 重力加速度 [m/s²]

    @property
    def M_total(self) -> float:
        return self.M_b + self.M_w


@dataclass
class ControllerParams:
    # --- 外側ループ: Δθ → α_ref [rad/s²] ---
    outer_Kp: float = 200.0
    outer_Ki: float = 0.0
    outer_Kd: float = 20.0
    alpha_max: float = 100.0   # 角加速度指令の上限 [rad/s²]

    # --- 内側ループ: Δω → PWM ---
    inner_Kp: float = 8.0
    inner_Ki: float = 80.0
    pwm_max: float = 255.0
