"""
Основной скрипт симуляции перевёрнутого маятника с PID-регулятором.

Содержит конфигурации маятника, датчиков, шумов и регулятора.
Запускает pygame-визуализацию с управлением от PID.
"""

from __future__ import annotations

import numpy as np
from optimizers import Zigler_Nikols, Genetic_PID_AngleOnly
from loggers import Logger

from packages.simulation.CO import (
    ControllerConfig,
    NoiseForce,
    ObjectOfControl,
    PlantConfig,
    SensorConfig,
)
from packages.controllers.PID import PIDController, terminate_condition
from packages.simulation.GUI import PendulumViewer


# ═══════════════════════════════════════════════════════════════════════════
# Конфигурация маятника (PlantConfig)
# ═══════════════════════════════════════════════════════════════════════════

PLANT_CONFIG = PlantConfig(
    M=1.0,          # масса тележки, кг
    m1=0.3,         # масса маятника, кг
    m2=0.0,         # второе звено включено
    l1=1.0,         # длина маятника, м
    l2=1.0,
    L1=0.7,         # расстояние до ЦМ маятника, м
    L2=0.0,
    J1=0.02,        # момент инерции маятника, кг·м²
    J2=0.00,
    g=-9.81,         # ускорение свободного падения, м/с²
    b_c=0.05,       # вязкое трение тележки
    b_1=0.05,      # вязкое трение в шарнире
    b_2=0.00,
    single_pendulum_mode=True,   # двухзвенный режим
    backslash_mode=False,        # люфт выключен
    init_q=np.array([0.0, np.pi, 0.0]),   # маятник вверху
    init_dq=np.array([0.0, 0.0, 0.0]),
    dt=0.005
)

# ═══════════════════════════════════════════════════════════════════════════
# Конфигурация датчиков и шумов (SensorConfig)
# ═══════════════════════════════════════════════════════════════════════════

SENSOR_CONFIG = SensorConfig(
    encoder_resolution_1=4096,     # 12 бит
    encoder_resolution_2=4096,     # 12 бит
    cart_sensor_resolution=0.0001, # шаг оптической линейки, м
    noise_std_q=(0.001, 0.005, 0.005),   # СКО шума координат
    noise_std_dq=(0.01, 0.02, 0.02),     # СКО шума скоростей
)

# ═══════════════════════════════════════════════════════════════════════════
# Конфигурация PID-регулятора (ControllerConfig)
# ═══════════════════════════════════════════════════════════════════════════

CONTROLLER_CONFIG = ControllerConfig(
    dt=0.005,                    # такт УУ 200 Гц
    max_force=30.0,              # макс. сила мотора, Н
    has_velocity_sensors=False,  
    differentiator_cutoff_hz=20.0, # фильтрация дифференциатора
    filter_cutoff_hz=10.0,         # фильтрация сигнала
    gains=   [25.27, 20, 3.99, -1, -0.1] # [Kp, Ki, Kd, Kx, Kdx]
)

# Инициализация контроллера
controller = PIDController(CONTROLLER_CONFIG)
controller.set_motor_inertia(time_constant=0.1)
method_options={
    "early_stop_angle": 0.01,
    "early_stop_x": 0.02,
    "early_stop_steps": 50,
    "K_p_range": [0.0, 70.0],
    "K_d_range": [0.0, 10.0],
    "K_x_range": [-15.0, -1.0],      # ← сузил, чтобы Kx не был слишком маленьким
    "K_dx_range": [-15.0, 0.0],
    "K_i_range": [0.0, 0.5],        # ← уменьшил интеграл
    "population_size": 50,
    "generations": 30,
    "top_k": 6,
    "final_error_weight": 1000.0,
    "overshoot_weight": 100.0,      # ← меньше штрафа за перелёт
    "integral_weight": 2.0,         # ← больше штрафа за ошибку
    "settling_weight": 50.0,        # ← главный приоритет
    "late_brake_weight": 10.0,
}
# controller.train(
#     PLANT_CONFIG, 
#     SENSOR_CONFIG, 
#     NoiseForce(mean=0.05, std=0.01),
#     target_state=np.array([-3, np.pi, 0, 0,0,0]),
#     terminate_condition=terminate_condition,
#     episode_max_time=1000.0,
#     logger=Logger(),
#     optimizer=Genetic_PID_AngleOnly(),
#     method_options=method_options
# )

# Инициализация объекта управления
plant = ObjectOfControl(PLANT_CONFIG)
# 

# ═══════════════════════════════════════════════════════════════════════════
# Запуск симуляции
# ═══════════════════════════════════════════════════════════════════════════

window = PendulumViewer(
    plant,
    SENSOR_CONFIG,
    NoiseForce(mean=0.05, std=0.02),
    controller=controller,
    # terminate_condition=terminate_condition,
    target_state=np.array([0, np.pi, 0, 0, 0, 0]),
)
window.use()
