"""Профилирование обучения RL-агента (REINFORCE) на 3 эпизода.

Запуск:
    poetry run python profiling/profile_reinforce_train.py

Результаты:
    - profiling_outputs/*.pstats (cProfile)
"""

from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import time
import cProfile
import pstats

from packages.controllers.REINFORCE.reinforce import Reinforce
from packages.controllers.REINFORCE.mode_config import ReinforceNetworkConfig
from packages.simulation.CO.datatypes import (
    ControllerConfig,
    PlantConfig,
    SensorConfig,
    NoiseForce,
)


def main() -> None:
    out_dir = Path("profiling_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    net_cfg = ReinforceNetworkConfig(
        state_dim=12,
        action_dim=1,
        hidden_layers=[16,16],
        activation="relu",
        learning_rate=1e-4,
        output_activation="tanh",
    )
    controller_cfg = ControllerConfig(dt=0.01, max_force=30.0)
    agent = Reinforce(net_cfg, controller_cfg)

    plant_cfg = PlantConfig(dt=0.001)
    sensor_cfg = SensorConfig()
    noise = NoiseForce(mean=0.0, std=0.0)
    target = np.array([0.0, np.pi, 0.0, 0.0, 0.0, 0.0])

    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    profiler.enable()
    try:
        agent.train(
            plant_config=plant_cfg,
            sensor_config=sensor_cfg,
            noise=noise,
            target_state=target,
            episode_max_time=10.0,
            episodes=3,
        )
    finally:
        profiler.disable()
    elapsed = time.perf_counter() - t0

    print(f"\nREINFORCE training (3 episodes) finished in {elapsed:.3f} sec")

    pstats_path = out_dir / f"reinforce_train_{int(t0)}.pstats"
    profiler.dump_stats(str(pstats_path))

    ps = pstats.Stats(str(pstats_path))
    ps.strip_dirs().sort_stats("cumtime").print_stats(30)


if __name__ == "__main__":
    main()