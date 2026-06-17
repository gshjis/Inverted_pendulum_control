from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

import packages.controllers.PID.cost_functions as cf
from loggers import Logger
from packages.controllers.PID.pid import PIDController
from packages.simulation.CO.datatypes import NoiseForce
from packages.simulation.CO.pendulum import ObjectOfControl
from packages.simulation.CO.run import clock_cycle
from packages.simulation.CO.sensor import SensorBlock


# ═══════════════════════════════════════════════════════════════════════════
#  ЦИГЛЕР-НИКОЛС ДЛЯ ПОЛОЖЕНИЯ (Kx, Kdx)
# ═══════════════════════════════════════════════════════════════════════════

class Zigler_Nikols:
    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger

    def optimize(
        self,
        controller: PIDController,
        plant: ObjectOfControl,
        sensor: SensorBlock,
        noise: NoiseForce,
        target_state: np.ndarray,
        terminate_condition: Callable[[ObjectOfControl], bool],
        episode_max_time: float,
        logger: Optional[Logger] = None,
        **kwargs,
    ):
        logger = logger or self.logger

        # Фиксируем угловые коэффициенты (уже настроены GA)
        fixed_Kp = float(kwargs.get("fixed_Kp", controller.gains[0]))
        fixed_Ki = float(kwargs.get("fixed_Ki", 0.0))
        fixed_Kd = float(kwargs.get("fixed_Kd", controller.gains[2]))

        # Диапазон для Kx
        Kx_range = kwargs.get("Kx_range", [-40.0, -1.0])
        Kx_step = kwargs.get("Kx_step", -0.5)
        Kx_min, Kx_max = float(Kx_range[0]), float(Kx_range[1])

        max_steps = int(episode_max_time / controller._dt)

        controller.gains = np.array([fixed_Kp, fixed_Ki, fixed_Kd, 0.0, 0.0], dtype=float)

        Kx_crit = None
        T_crit = None
        Kx = 0.0

        if logger:
            print("[Zigler-Nikols] Поиск Kx_crit...")

        while Kx >= Kx_min and Kx_crit is None:
            controller.gains[3] = Kx
            controller.gains[4] = 0.0

            plant.reset()
            controller.reset()
            plant.q[0] = 3.0
            F = 0.0

            trajectory = np.zeros(max_steps)
            steps_done = 0

            for step in range(max_steps):
                _, F = clock_cycle(controller, plant, sensor, noise, F, target_state, cf.J)
                if terminate_condition(plant):
                    break
                trajectory[step] = plant.q[0]
                steps_done = step + 1

            if logger:
                logger.draw_dynamic_plot(trajectory[:steps_done], plant._dt)

            is_osc, period = self._detect_oscillations(trajectory[:steps_done], plant._dt)

            if is_osc:
                Kx_crit = Kx
                T_crit = period
                print(f"    ✅ Kx_crit = {Kx_crit:.2f}, T_crit = {T_crit:.3f} с")
            else:
                print(f"    Kx = {Kx:.2f}: нет колебаний")
                Kx += Kx_step

        if Kx_crit is None:
            raise RuntimeError("Kx_crit не найден")

        # Формулы Циглера-Николса для ПД
        Kx_final = 0.6 * Kx_crit
        Kdx_final = 0.125 * Kx_crit * T_crit

        controller.gains = np.array([fixed_Kp, fixed_Ki, fixed_Kd, Kx_final, Kdx_final], dtype=float)

        print(f"[Zigler-Nikols] ✅ Kx = {Kx_final:.4f}, Kdx = {Kdx_final:.4f}")

        return {
            "Kx_crit": float(Kx_crit),
            "T_crit": float(T_crit),
            "Kx_final": float(Kx_final),
            "Kdx_final": float(Kdx_final),
        }

    def _detect_oscillations(self, trajectory: np.ndarray, dt: float) -> tuple[bool, float]:
        from scipy.signal import find_peaks

        if len(trajectory) < 100:
            return False, 0.0

        start = int(0.3 * len(trajectory))
        sig = trajectory[start:]

        peaks, _ = find_peaks(sig, height=0.01)
        if len(peaks) < 6:
            return False, 0.0

        peaks = peaks[-10:]
        peak_values = sig[peaks]

        half = len(peaks) // 2
        if half < 2:
            return False, 0.0

        mean_first = np.mean(peak_values[:half])
        mean_last = np.mean(peak_values[half:])

        if abs(mean_last - mean_first) / max(mean_first, 0.001) > 0.05:
            return False, 0.0

        periods = np.diff(peaks) * dt
        return True, float(np.mean(periods))


# ═══════════════════════════════════════════════════════════════════════════
#  GA ДЛЯ УГЛА (Kp, Ki, Kd)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class _GA2Config:
    population_size: int = 24
    generations: int = 40
    elite_frac: float = 0.2
    tournament_k: int = 3
    mutation_sigma: float = 1.5
    mutation_prob: float = 0.4
    crossover_prob: float = 0.7
    seed: int | None = None


class Genetic_PID_AngleOnly:
    def __init__(self, logger: Optional[Logger] = None):
        self._logger = logger

    def optimize(
        self,
        controller: PIDController,
        plant: ObjectOfControl,
        sensor: SensorBlock,
        noise: NoiseForce,
        target_state: np.ndarray|Callable,
        terminate_condition: Callable[[ObjectOfControl], bool],
        episode_max_time: float,
        logger: Optional[Logger] = None,
        **kwargs,
    ):
        logger = logger or self._logger

        fixed_Kx = float(kwargs.get("fixed_Kx", 0.0))
        fixed_Kdx = float(kwargs.get("fixed_Kdx", 0.0))

        Kp_range = kwargs.get("Kp_range", [0.0, 70.0])
        Ki_range = kwargs.get("Ki_range", [0.0, 50.0])
        Kd_range = kwargs.get("Kd_range", [0.0, 10.0])
        Kp_min, Kp_max = float(Kp_range[0]), float(Kp_range[1])
        Ki_min, Ki_max = float(Ki_range[0]), float(Ki_range[1])
        Kd_min, Kd_max = float(Kd_range[0]), float(Kd_range[1])

        ga = _GA2Config(
            population_size=int(kwargs.get("population_size", 50)),
            generations=int(kwargs.get("generations", 30)),
            elite_frac=float(kwargs.get("elite_frac", 0.2)),
            tournament_k=int(kwargs.get("tournament_k", 3)),
            mutation_sigma=float(kwargs.get("mutation_sigma", 1.5)),
            mutation_prob=float(kwargs.get("mutation_prob", 0.4)),
            crossover_prob=float(kwargs.get("crossover_prob", 0.7)),
            seed=kwargs.get("seed", None),
        )

        rng = np.random.default_rng(ga.seed)
        max_steps = int(episode_max_time / controller._dt)
        angle_goal = float(getattr(target_state, "y", target_state[1]))
        early_stop_angle = float(kwargs.get("early_stop_angle", 0.01))
        early_stop_steps = int(kwargs.get("early_stop_steps", 50))

        def _set_gains(Kp: float, Ki: float, Kd: float) -> None:
            controller.gains = np.array([Kp, Ki, Kd, fixed_Kx, fixed_Kdx], dtype=float)

        # ─── Fitness ──────────────────────────────────────────────────────
        def fitness_hold(Kp: float, Ki: float, Kd: float) -> float:
            _set_gains(Kp, Ki, Kd)
            plant.reset()
            controller.reset()
            F = 0.0
            stable_counter = 0

            for step in range(max_steps):
                _, F = clock_cycle(controller, plant, sensor, noise, F, target_state, cf.J)
                if terminate_condition(plant):
                    return 1e6 + float(step)

                if abs(plant.q[1] - angle_goal) < early_stop_angle:
                    stable_counter += 1
                    if stable_counter >= early_stop_steps:
                        return 0.0
                else:
                    stable_counter = 0
            return 0.0

        # ─── Популяция ──────────────────────────────────────────────────────
        pop = rng.uniform(
            low=[Kp_min, Ki_min, Kd_min],
            high=[Kp_max, Ki_max, Kd_max],
            size=(ga.population_size, 3),
        )
        best_params = pop[0].copy()
        best_fit = float("inf")
        best_hold_params = None
        best_hold_fit = float("inf")

        for gen in range(ga.generations):
            fits_hold = np.array([
                fitness_hold(float(x[0]), float(x[1]), float(x[2])) for x in pop
            ], dtype=float)

            gen_best_idx = int(np.argmin(fits_hold))
            gen_best = pop[gen_best_idx].copy()
            gen_best_fit = float(fits_hold[gen_best_idx])

            if gen_best_fit < best_fit:
                best_fit = gen_best_fit
                best_params = gen_best

            if gen_best_fit < best_hold_fit:
                best_hold_fit = gen_best_fit
                best_hold_params = gen_best.copy()

            if logger:
                print(
                    f"[GA] gen={gen+1}/{ga.generations} "
                    f"Kp={gen_best[0]:.2f} Ki={gen_best[1]:.2f} Kd={gen_best[2]:.2f} "
                    f"hold={gen_best_fit}"
                )

            # ─── Элитный отбор ──────────────────────────────────────────
            n_elite = max(1, int(round(ga.elite_frac * ga.population_size)))
            elite_idx = np.argsort(fits_hold)[:n_elite]
            elite = pop[elite_idx]

            new_pop = [elite[i].copy() for i in range(n_elite)]
            while len(new_pop) < ga.population_size:
                idx1 = rng.integers(0, ga.population_size, size=ga.tournament_k)
                p1 = pop[idx1[int(np.argmin(fits_hold[idx1]))]]
                idx2 = rng.integers(0, ga.population_size, size=ga.tournament_k)
                p2 = pop[idx2[int(np.argmin(fits_hold[idx2]))]]

                child = p1.copy()
                if rng.random() < ga.crossover_prob:
                    alpha = rng.random()
                    child = alpha * p1 + (1.0 - alpha) * p2

                if rng.random() < ga.mutation_prob:
                    sigma = ga.mutation_sigma * (1.0 - gen / max(1, ga.generations - 1))
                    child = child + rng.normal(0.0, sigma, size=child.shape)

                child[0] = float(np.clip(child[0], Kp_min, Kp_max))
                child[1] = float(np.clip(child[1], Ki_min, Ki_max))
                child[2] = float(np.clip(child[2], Kd_min, Kd_max))
                new_pop.append(child)

            pop = np.asarray(new_pop, dtype=float)

        if best_hold_params is not None:
            _set_gains(float(best_hold_params[0]), float(best_hold_params[1]), float(best_hold_params[2]))

        print(
            f"[GA] ✅ Kp={controller.gains[0]:.4f}, "
            f"Ki={controller.gains[1]:.4f}, "
            f"Kd={controller.gains[2]:.4f}"
        )

        return {
            "best_Kp": float(controller.gains[0]),
            "best_Ki": float(controller.gains[1]),
            "best_Kd": float(controller.gains[2]),
        }
