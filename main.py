"""
Основной скрипт симуляции перевёрнутого маятника с PID-регулятором.

Содержит конфигурации маятника, датчиков, шумов и регулятора.
Запускает pygame-визуализацию с управлением от PID.
"""

from __future__ import annotations

import numpy as np

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
    init_q=np.array([3.0, np.pi, 0.0]),   # маятник вверху
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
    gains=  [118.99, 0.23, 80.08, -5.00, -13.87]  # [Kp, Ki, Kd, Kx, Kdx]
)

# Инициализация контроллера
controller = PIDController(CONTROLLER_CONFIG)
# Добавляем инерционность двигателя (tau = 0.05 с)
controller.set_motor_inertia(time_constant=0.05)
# controller.train(
#     PLANT_CONFIG, 
#     SENSOR_CONFIG, 
#     NoiseForce(mean=0.05, std=0.01),
#     target_state=MeasuredState(0, np.pi, 0),
#     terminate_condition=terminate_condition,
#     max_time=40.0,
#     method_options={
#         "generations": 15,
#         "population_size": 100,
#         "mutation_sigma": 0.15,
#         "mutation_prob": 0.4,
#         "early_stop_threshold": 10.0,  # остановиться при J < 10
#     }
# )
# Инициализация объекта управления
plant = ObjectOfControl(PLANT_CONFIG)

# ═══════════════════════════════════════════════════════════════════════════
# Запуск симуляции
# ═══════════════════════════════════════════════════════════════════════════

window = PendulumViewer(
    plant,
    SENSOR_CONFIG,
    NoiseForce(mean=0.05, std=0.02),
    # controller=controller,
    # terminate_condition=terminate_condition,
    target_state=np.array([0, np.pi, 0]),
)
window.use()
