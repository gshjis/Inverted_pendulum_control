"""
Основной скрипт симуляции перевёрнутого маятника.
Запускает Pygame-визуализацию с PID-регулятором и SwingUp-раскачкой.
"""

from __future__ import annotations

import numpy as np

from packages.controllers.PID import PIDController
from packages.controllers.custom import SwingUp
from packages.controllers.custom.swing_up_block import SwingUpAndBalance
from packages.simulation.CO import (
    ControllerConfig,
    NoiseForce,
    ObjectOfControl,
    PlantConfig,
    SensorConfig,
)
from packages.simulation.GUI import PendulumViewer

# ═══════════════════════════════════════════════════════════════════════════
# Конфигурация физической модели (тележка + двухзвенный маятник)
# ═══════════════════════════════════════════════════════════════════════════

PLANT_CONFIG = PlantConfig(
    # === Тележка ===
    M=1.0,

    # === Нижнее звено ===
    m1=0.1,
    l1=0.3,
    b_1=0.003,

    # === Верхнее звено ===
    m2=0.1,
    l2=0.3,
    b_2=0.003,

    # === Общие ===
    g=-9.81,
    b_c=0.1,

    # === Режимы ===
    single_pendulum_mode=False,
    backslash_mode=False,

    # === Начальное состояние ===
    init_q=np.array([0.0, np.pi, 0.0]),
    init_dq=np.array([0.0, 0.0, 0.0]),
    dt=0.0001,
)

# ═══════════════════════════════════════════════════════════════════════════
# Конфигурация датчиков
# ═══════════════════════════════════════════════════════════════════════════

SENSOR_CONFIG = SensorConfig(
    encoder_resolution_1=4096,     # 12 бит — 4096 отсчётов на оборот
    encoder_resolution_2=4096,
    cart_sensor_resolution=0.0001, # 0.1 мм
    noise_std_q=(0.0005, 0.002, 0.002),   # ~0.03° по углам
    noise_std_dq=(0.005, 0.01, 0.01),     # скорости
)

# ═══════════════════════════════════════════════════════════════════════════
# Конфигурация контроллера
# ═══════════════════════════════════════════════════════════════════════════

CONTROLLER_CONFIG = ControllerConfig(
    dt=0.001,
    max_force=24,
    has_velocity_sensors=True,
    filter_cutoff_hz=50.0,
)

# ═══════════════════════════════════════════════════════════════════════════
# Точка входа
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # Контроллеры
    swing_controller = SwingUp(CONTROLLER_CONFIG, K=150, plant_config=PLANT_CONFIG)
    pid_controller = PIDController(
        CONTROLLER_CONFIG,
        gains=np.array([80.42, 0.0, 30.71, -10, -15]),
    )
    controller = SwingUpAndBalance(
        CONTROLLER_CONFIG,
        swingup_controller=swing_controller,
        balance_controller=pid_controller,
    )
    controller.set_motor_inertia(time_constant=0.001)

    # Внешнее возмущение и целевое состояние
    NOISE = NoiseForce(mean=0.00, std=0.03)
    TARGET = np.array([0.0, np.pi, 0.0, 0.0, 0.0, 0.0])  # (x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂)

    # Запуск визуализации
    viewer = PendulumViewer(
        plant=ObjectOfControl(PLANT_CONFIG),
        sensor_config=SENSOR_CONFIG,
        noise=NOISE,
        target_state=TARGET,
        controller=controller,
    )
    viewer.use()
