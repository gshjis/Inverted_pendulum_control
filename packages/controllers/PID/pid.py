from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from packages.simulation.CO import (
    Controller,
    ControllerConfig,
    MeasuredState,
    NoiseForce,
    ObjectOfControl,
    PlantConfig,
    SensorBlock,
    SensorConfig,
)


class PIDController(Controller):
    def __init__(self, config: ControllerConfig) -> None:
        super().__init__(config)
        self.name = "PID"

        if isinstance(config, ControllerConfig):
            gains = config.gains
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

    def get_action(self, s_clean: MeasuredState, target_state:MeasuredState) -> float:

        error = target_state - s_clean

        self._integral += error.theta1 * self._dt

        F = (
            self._Kp * error.theta1
            + self._Ki * self._integral
            + self._Kd * error.theta1_dot
            + self._Kx * error.x
            + self._Kdx * error.x_dot
        )
        return F

    def reset_angel_integral(self) -> None:
        self._integral = 0.0

    # ── Сброс ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        super().reset()
        self._integral = 0.0

    # ── Обучение ──────────────────────────────────────────────────────────

    def train(
        self,
        plant_config: PlantConfig,
        sensor_config: SensorConfig,
        noise: NoiseForce,
        *,
        alpha: float = 1.0,
        max_time: float = 10.0,
        method_options: dict[str, Any] | None = None,
        target_state: MeasuredState,
        terminate_condition: Callable[[ObjectOfControl], bool] | None = None,
    ) -> dict[str, Any]:
        # Более исследовательский метод оптимизации (стохастический поиск).
        # Реализуем простую эволюционную стратегию: на каждой итерации
        # генерируем партию кандидатов вокруг текущих параметров, оцениваем
        # их и выбираем лучший; шаг (sigma) уменьшаем по расписанию.
        rng = np.random.default_rng()
        current = self.gains.copy()
        best_J = float("inf")
        best_x = current.copy()

        max_iters = (method_options or {}).get("maxiter", 200)
        pop_size = (method_options or {}).get("pop_size", 16)
        sigma0 = (method_options or {}).get("sigma", 1.0)
        sigma = float(sigma0)

        for it in range(int(max_iters)):
            # уменьшение дисперсии по экспоненциальной схеме
            sigma = sigma0 * (0.99 ** it)

            # сгенерировать популяцию кандидатов
            candidates = current + rng.normal(scale=sigma, size=(pop_size, current.size))

            scores = np.zeros(pop_size, dtype=np.float64)
            times = np.zeros(pop_size, dtype=np.float64)
            for i in range(pop_size):
                self.gains = candidates[i]
                J, t = self._run_episode(
                    plant_config, sensor_config, noise, alpha, max_time, target_state, terminate_condition
                )
                scores[i] = J
                times[i] = t

            # Найти лучшего в популяции
            idx = int(np.argmin(scores))
            if scores[idx] < best_J:
                best_J = float(scores[idx])
                best_x = candidates[idx].copy()

            # Логирование прогресса
            avg_J = float(np.mean(scores))
            print(
                f"[it {it:04d}] best={best_J:8.4f} avg={avg_J:8.4f} "
                f"Kp={best_x[0]:.4f} Ki={best_x[1]:.4f} Kd={best_x[2]:.4f} "
                f"Kx={best_x[3]:.4f} Kdx={best_x[4]:.4f} sigma={sigma:.4f}"
            )

            # Сдвинуть центр к лучшему кандидату
            current = best_x.copy()

            # простое условие останова — если улучшение мало
            if it > 5 and abs(avg_J - best_J) < 1e-6:
                break

        # Применить лучшие найденные параметры
        self.gains = best_x
        return {"x": best_x, "fun": best_J, "success": True}

    # ── Внутренний метод: прогон эпизода ──────────────────────────────────

    def _run_episode(
        self,
        plant_config: PlantConfig,
        sensor_config: SensorConfig,
        noise: NoiseForce,
        alpha: float,
        max_time: float,
        target_state: MeasuredState,
        terminate_condition: Callable[[ObjectOfControl], bool] | None = None,
    ) -> tuple[float, float]:
        plant = ObjectOfControl(plant_config)
        sensor = SensorBlock(sensor_config)

        dt_control = self._dt
        dt_physics = 0.0005
        steps_per_control = int(dt_control / dt_physics)

        max_steps = int(max_time / dt_control)
        J = 0.0
        step = 0

        self.reset()

        for step in range(max_steps):
            measured = sensor.get_telemetry(plant.q, plant.dq)

            # проверка терминального состояния (пользовательская или дефолтная)
            terminated = False
            if terminate_condition is not None:
                terminated = bool(terminate_condition(plant))
            else:
                if abs(measured.theta1 - np.pi) > np.radians(15.0) or abs(measured.x) > 4:
                    terminated = True

            if terminated:
                J += (max_steps - step) * 4.0
                break

            F_raw = self.compute_control(measured, target_state)

            for _ in range(steps_per_control):
                plant.update_physics(F_raw, noise, dt_physics)
            
            th = plant.q[1]
            x_pos = plant.q[0]
            J += ((th - target_state.theta1) ** 2 + alpha * x_pos ** 2) * dt_control

        return (J, step*dt_control)
