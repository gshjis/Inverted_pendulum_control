"""
Основной скрипт симуляции перевёрнутого маятника с PID-регулятором.
"""

from __future__ import annotations

import numpy as np
from pid import PIDController,terminate_condition
from swing_up_block import SwingUp, SwingUpAndBalance

from packages.controllers.custom import SwingUp
from packages.controllers.PID.optimizers import Genetic_PID_AngleOnly, Zigler_Nikols
from packages.simulation.CO import (
    ControllerConfig,
    NoiseForce,
    ObjectOfControl,
    PlantConfig,
    SensorConfig,
)
from packages.simulation.GUI import PendulumViewer

PLANT_CONFIG = PlantConfig(
    M=0.8,          # масса тележки, кг
    m1=0.25,        # масса маятника, кг
    m2=0.0,
    l1=1.1,         # длина маятника, м
    l2=0.0,
    L1=0.7,         # расстояние до ЦМ маятника, м
    L2=0.0,
    J1=0.015,       # момент инерции маятника, кг·м²
    J2=0.00,
    g=-9.81,        # ускорение свободного падения, м/с²
    b_c=0.05,        # вязкое трение тележки
    b_1=0.01,        # вязкое трение в шарнире
    b_2=0.00,
    single_pendulum_mode=True,
    backslash_mode=False,
    init_q=np.array([0.0, 0.0, 0.0]),  # маятник в верхнем положении
    init_dq=np.array([0.0, 0.0, 0.0]),
    dt=0.0001,
)

SENSOR_CONFIG = SensorConfig(
    encoder_resolution_1=4096,     # 14 бит — 16384 отсчёта на оборот (~0.022°)
    encoder_resolution_2=4096,
    cart_sensor_resolution=0.0001, # 0.05 мм
    noise_std_q=(0.0005, 0.002, 0.002),   # ~0.03° по углам
    noise_std_dq=(0.005, 0.01, 0.01),     # скорости
)

CONTROLLER_CONFIG = ControllerConfig(
    dt=0.01,
    max_force=40.0,
    has_velocity_sensors=True,
    filter_cutoff_hz=50.0,
)


if __name__ == "__main__":

    swing_controller = SwingUp(CONTROLLER_CONFIG,30,PLANT_CONFIG)
    pid_controller = PIDController(CONTROLLER_CONFIG, gains=np.array([66.9, 0,9.8,-6,-7.2]))
    controller = SwingUpAndBalance(
        CONTROLLER_CONFIG,
        swingup_controller=swing_controller,
        balance_controller=pid_controller
    )
    controller.set_motor_inertia(time_constant=0.01)

    NOISE = NoiseForce(mean=0.005, std=0.1)
    TARGET = np.array([0.0, np.pi, 0.0, 0.0, 0.0, 0.0])

    optimizer = Genetic_PID_AngleOnly()
    # pid_controller.train(
    #     plant_config=PLANT_CONFIG,
    #     sensor_config=SENSOR_CONFIG,
    #     noise=NOISE,
    #     optimizer=optimizer,
    #     target_state=TARGET,
    #     episode_max_time=30.0,
    #     terminate_condition=terminate_condition
    # )

    w = PendulumViewer(
        plant=ObjectOfControl(PLANT_CONFIG),
        sensor_config=SENSOR_CONFIG,
        noise=NOISE,
        target_state=TARGET,
        controller=controller,
    )
    w.use()
