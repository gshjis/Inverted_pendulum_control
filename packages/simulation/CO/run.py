from packages.simulation.CO.datatypes import NoiseForce
from packages.simulation.CO.pendulum import ObjectOfControl
from packages.simulation.CO.sensor import SensorBlock
from packages.simulation.CO.controller import Controller
from typing import Callable
import numpy as np


def clock_cycle(
    controller: Controller,
    plant: ObjectOfControl,
    sensor: SensorBlock,
    noise: NoiseForce,
    old_F: float,
    target_state: np.ndarray,
    J: Callable[[np.ndarray, np.ndarray], float],
) -> tuple[float, float]:
    """Выполнить один такт управления (control-tick) и вернуть значение целевой функции.

    Логика включает искусственную задержку вычисления управляющего воздействия:
    управление пересчитывается только после того, как физика отработала ~75%
    от длительности control-tick, используя предыдущее значение силы.

    Parameters
    ----------
    controller:
        Блок вычисления управления.
    plant:
        Объект управления (физическая модель).
    sensor:
        Блок получения телеметрии.
    noise:
        Мгновенное внешнее возмущение.
    old_F:
        Сила, которую применяют к объекту управления до момента пересчёта
        управляющего сигнала.
    target_state:
        Целевое (измеряемое) состояние.
    J:
        Функция стоимости/оценки качества: ``J(measured, target_state)``.

    Returns
    -------
    float
        Значение ``J`` на последней телеметрии в пределах одного control-tick.
    """

    dt_control = controller._dt
    dt_physics = plant._dt

    # Общее число шагов физики в одном тикe управления.
    steps_per_control = int(dt_control / dt_physics)
    # Доля времени, в течение которой управление не пересчитывается (задержка).
    freeze_ratio = 0.75
    count_time = int(steps_per_control * freeze_ratio)
    remaining_steps = steps_per_control - count_time

    F_raw = float(old_F)
    measured = sensor.get_telemetry(plant.q, plant.dq)

    # Фаза задержки: физика работает, используя предыдущее воздействие.
    for _ in range(count_time):
        plant.update_physics(F_raw, noise)

    # Пересчёт управления.
    F_raw = controller.compute_control(measured, target_state)

    # Фаза применения нового управления до конца control-tick.
    for _ in range(remaining_steps):
        plant.update_physics(F_raw, noise)

    measured = sensor.get_telemetry(plant.q, plant.dq)
    return J(target_state, measured), F_raw
