"""Профилирование метода train у PID-контроллера.

Запуск:
    poetry run python profiling/profile_pid_train.py

Результаты:
    - profiling_outputs/*.pstats (cProfile)
    - profiling_outputs/*.json (line_profiler при наличии)

Важно:
Этот скрипт не пытается воспроизвести обучение до идеального результата.
Он делает несколько итераций train, чтобы увидеть узкие места.
"""

from __future__ import annotations

import os
import numpy as np
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import time
import cProfile
import pstats

from pendulum import ObjectOfControl
from sensor import SensorBlock



from packages.controllers.PID import PIDController  # noqa: E402
from packages.simulation.CO.datatypes import (  # noqa: E402
    ControllerConfig,
    PlantConfig,
    SensorConfig,
    NoiseForce,
)

def main() -> None:
    out_dir = Path("profiling_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Берём конфигурации “по умолчанию”, чтобы скрипт работал сразу.
    controller_cfg = ControllerConfig(dt=0.005, gains=[10.0, 1.0, 2.0, 1.0, 2.0])
    pid = PIDController(controller_cfg)
    pid.set_motor_inertia(0.05)

    plant_cfg = PlantConfig(dt=0.005)
    sensor_cfg = SensorConfig()
    noise = NoiseForce(mean=0.0, std=0.0)

    target = np.array([0.0,  3.141592653589793,0.0,  0.0,    0.0,    0.0]    )

    # Ограничиваем размер задачи, чтобы профиль было реально получить.

    plant = ObjectOfControl(plant_cfg)
    sensor = SensorBlock(sensor_cfg)
    # Общий профиль CPU времени.
    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    profiler.enable()
    try:
        for i in range(100_000):
            F_id = pid.compute_control(target, target)
            plant.update_physics(F_id, noise)
            sensor.get_telemetry(plant.q, plant.dq)
            
    finally:
        profiler.disable()
    elapsed = time.perf_counter() - t0

    print("Simulation finished in", elapsed, "sec")

    pstats_path = out_dir / f"pid_train_{int(t0)}.pstats"
    profiler.dump_stats(str(pstats_path))

    # Топ по времени
    ps = pstats.Stats(str(pstats_path))
    ps.strip_dirs().sort_stats("cumtime").print_stats(30)


if __name__ == "__main__":
    main()
