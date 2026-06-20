from __future__ import annotations

from typing import Any, Callable, Optional
import cost_functions as cost_f
from loggers import Logger
import numpy as np
from numpy.typing import NDArray
from packages.simulation.CO import (
    Controller,
    ControllerConfig,
    ObjectOfControl,
    PlantConfig,
    SensorBlock,
    SensorConfig,
    clock_cycle,
    NoiseForce
)

def terminate_condition(state:ObjectOfControl) -> bool:
    if abs(state.q[1] - np.pi) > np.radians(40):
        return False
    return False



class PIDController(Controller):
    def __init__(self, config: ControllerConfig, gains: np.ndarray | None = None) -> None:
        super().__init__(config)
        self.name = "PID"

        if gains is None:
            gains = np.array([10.0, 1.0, 2.0, 1.0, 2.0], dtype=np.float64)
        self._Kp: float = float(gains[0])
        self._Ki: float = float(gains[1])
        self._Kd: float = float(gains[2])
        self._Kx: float = float(gains[3])
        self._Kdx: float = float(gains[4])

        self._integral: float = 0.0

    # ── Свойства ──────────────────────────────────────────────────────────

    @property
    def gains(self) -> NDArray[np.float64]:
        return np.array([self._Kp, self._Ki, self._Kd, self._Kx, self._Kdx], dtype=np.float64)

    @gains.setter
    def gains(self, value: list[float] | NDArray[np.float64]) -> None:
        self._Kp, self._Ki, self._Kd, self._Kx, self._Kdx = map(float, value)

    # ── Закон управления ──────────────────────────────────────────────────

    def get_action(self, s_clean: np.ndarray, target_state:np.ndarray) -> float:
        error = target_state - s_clean

        self._integral += error[1] * self._dt

        F = (
            self._Kp * error[1]
            + self._Ki * self._integral
            + self._Kd * error[4]
            + self._Kx * error[0]
            + self._Kdx * error[3]
        )
        return F

    def reset_angel_integral(self) -> None:
        self._integral = 0.0

    # ── Сброс ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        super().reset()
        self._integral = 0.0

    # ── Внутренний метод: прогон эпизода ──────────────────────────────────


    def _run_episode(
        self,
        plant: ObjectOfControl,
        sensor: SensorBlock,
        noise: NoiseForce,
        max_time: float,
        target_state: np.ndarray,
        terminate_condition: Callable[[ObjectOfControl], bool] | None = None,
    ) -> tuple[float, float]:

        dt_control = self._dt
        max_steps = int(max_time / dt_control)
        
        self.reset()
        F_raw = 0.0
        J = 0.0
        
        step = 0
        for step in range(max_steps):
            J_, F_raw = clock_cycle(self, plant, sensor, noise, F_raw, target_state, cost_f.J)
            if terminate_condition is not None and terminate_condition(plant):
                break
            J+= J_
        
        return float(J), (step + 1) * dt_control



    # ── Обучение ──────────────────────────────────────────────────────────

    def train(
        self,
        plant_config: PlantConfig,
        sensor_config: SensorConfig,
        noise: NoiseForce,
        optimizer,
        target_state: np.ndarray|Callable,
        terminate_condition: Callable[[ObjectOfControl], bool] | None = None,
        episode_max_time: float = 150.0,
        logger:Optional[Logger] = None,
        *,
        method_options: dict[str, Any] | None = None,
    ) -> None:
        plant = ObjectOfControl(plant_config)
        sensor = SensorBlock(sensor_config)
        a =optimizer.optimize(
            self,
            plant,
            sensor, 
            noise, 
            target_state, 
            terminate_condition, 
            episode_max_time,
            logger
        )
        print(a)
