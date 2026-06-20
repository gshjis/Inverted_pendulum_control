"""
Основной скрипт симуляции перевёрнутого маятника с PID-регулятором
и обучения RL-агента (REINFORCE).
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
from packages.simulation.GUI import PendulumViewer

from packages.controllers.REINFORCE.reinforce import Reinforce
from packages.controllers.REINFORCE.mode_config import ReinforceNetworkConfig


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


SENSOR_CONFIG = SensorConfig(
    encoder_resolution_1=4096,     # 12 бит
    encoder_resolution_2=4096,     # 12 бит
    cart_sensor_resolution=0.0001, # шаг оптической линейки, м
    noise_std_q=(0.001, 0.005, 0.005),   # СКО шума координат
    noise_std_dq=(0.01, 0.02, 0.02),     # СКО шума скоростей
)


# ═══════════════════════════════════════════════════════════════════════════
# REINFORCE — обучение
# ═══════════════════════════════════════════════════════════════════════════

NET_CONFIG = ReinforceNetworkConfig(
    state_dim=12,              # s_clean (6) + target_state (6)
    action_dim=1,              # одно управляющее воздействие — сила F
    hidden_layers=[128,128, 128],
    activation="gelu",
    learning_rate=1e-4,
    output_activation="gelu",
)

CONTROLLER_CONFIG = ControllerConfig(
    dt=0.05,
    max_force=30.0,
    has_velocity_sensors=True,
    filter_cutoff_hz=50.0,
)


if __name__ == "__main__":
    agent = Reinforce(NET_CONFIG, CONTROLLER_CONFIG)

    NOISE = NoiseForce(mean=0.05, std=0.1)
    TARGET = np.array([0.0, np.pi, 0.0, 0.0, 0.0, 0.0])
    agent.set_motor_inertia(time_constant=0.1)
    # agent.train(
    #     plant_config=PLANT_CONFIG,
    #     sensor_config=SENSOR_CONFIG,
    #     noise=NOISE,
    #     target_state=TARGET,
    #     episode_max_time=30.0,
    # )
    agent = Reinforce.from_pretrained(
        path="/home/gshjis/Python_projects/RL/checkpoints/reinforce/epoch_0420_return_+100.8_steps_588.pt",
        config=NET_CONFIG,
        controller_config=CONTROLLER_CONFIG,
    )
    w = PendulumViewer(
        plant=ObjectOfControl(PLANT_CONFIG),
        sensor_config=SENSOR_CONFIG,
        noise=NOISE,
        target_state=TARGET,
        controller=agent,
    )
    w.use()


