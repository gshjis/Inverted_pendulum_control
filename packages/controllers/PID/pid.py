from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from packages.simulation.CO import (
    Controller,
    ControllerConfig,
    MeasuredState,
    NoiseForce,
    ObjectOfControl,
    PlantConfig,
    SensorBlock,
    SensorConfig,
    State
)


class PIDController(Controller):
    def __init__(self, config: ControllerConfig) -> None:
        super().__init__(config)

        if isinstance(config, ControllerConfig):
            gains = config.gains
        self._Kp: float = float(gains[0])
        self._Ki: float = float(gains[1])
        self._Kd: float = float(gains[2])
        self._Kx: float = float(gains[3])

        self._integral: float = 0.0

    # ── Свойства ──────────────────────────────────────────────────────────

    @property
    def gains(self) -> NDArray[np.float64]:
        return np.array([self._Kp, self._Ki, self._Kd, self._Kx], dtype=np.float64)

    @gains.setter
    def gains(self, value: list[float] | NDArray[np.float64]) -> None:
        self._Kp, self._Ki, self._Kd, self._Kx = map(float, value)

    # ── Закон управления ──────────────────────────────────────────────────

    def get_action(self, s_clean: MeasuredState, target_state:State) -> float:
        x = target_state.x - s_clean.x
        th1 = target_state.theta1 - s_clean.theta1
        dth1 = -s_clean.theta1_dot

        self._integral += th1 * self._dt

        F = (
            self._Kp * th1
            + self._Ki * self._integral
            + self._Kd * dth1
            + self._Kx * x
        )
        return F

    # ── Сброс ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        super().reset()
        self._integral = 0.0

    # ── Обучение ──────────────────────────────────────────────────────────

    def train(
        self,
        plant_config: PlantConfig,
        sensor_config: SensorConfig,
        *,
        alpha: float = 1.0,
        max_time: float = 10.0,
        method_options: dict[str, Any] | None = None,
        target_state:State
    ) -> dict[str, Any]:
        _iteration = [0]

        def objective(gains_vector: NDArray[np.float64]) -> float:
            self.gains = gains_vector
            J, time = self._run_episode(
                self, plant_config, sensor_config, alpha, max_time,target_state
            )
            # Явное логгирование в консоль
            print(
                f"[iter {_iteration[0]:>3d}]  "
                f"J={J:>10.4f}  "
                f"Kp={gains_vector[0]:>8.4f}  "
                f"Ki={gains_vector[1]:>8.4f}  "
                f"Kd={gains_vector[2]:>8.4f}  "
                f"Kx={gains_vector[3]:>8.4f}  "
                f"t={time:>8.4f}"
            )
            _iteration[0] += 1
            return J

        x0 = self.gains
        options = method_options or {"xatol": 1e-4, "maxiter": 200}

        result = minimize(
            objective,
            x0,
            method="Nelder-Mead",
            options=options,
        )

        self.gains = result.x
        return {
            "x": result.x,
            "fun": result.fun,
            "success": result.success,
        }

    # ── Внутренний метод: прогон эпизода ──────────────────────────────────

    @staticmethod
    def _run_episode(
        controller: PIDController,
        plant_config: PlantConfig,
        sensor_config: SensorConfig,
        alpha: float,
        max_time: float,
        target_state: State
    ) -> tuple[float, float]:
        plant = ObjectOfControl(plant_config)
        sensor = SensorBlock(sensor_config)
        F_noise = NoiseForce(value=0.01)

        dt_control = 0.005
        dt_physics = 0.0005
        steps_per_control = int(dt_control / dt_physics)

        max_steps = int(max_time / dt_control)
        J = 0.0

        controller.reset()

        for step in range(max_steps):
            measured = sensor.get_telemetry(plant.q, plant.dq)

            if abs(measured[1] - np.pi) > np.radians(15.0) or abs(measured[0]) > 4:
                J += (max_steps - step) * 4.0
                break

            ms = MeasuredState(
                x=measured[0], theta1=measured[1], theta2=measured[2],
                x_dot=measured[3], theta1_dot=measured[4], theta2_dot=measured[5],
            )
            F_raw = controller.compute_control(ms, target_state)

            for _ in range(steps_per_control):
                plant.update_physics(F_raw, F_noise, dt_physics)

            th = plant.q[1]
            x_pos = plant.q[0]
            J += ((th - target_state.theta1) ** 2 + alpha * x_pos ** 2) * dt_control

        return (J, step*dt_control)
