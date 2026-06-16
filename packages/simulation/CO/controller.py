from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from .datatypes import (
    ControllerConfig,
)
from packages.simulation.CO.engine import MotorInertia


# ═══════════════════════════════════════════════════════════════════════════
# Differentiator
# ═══════════════════════════════════════════════════════════════════════════

class Differentiator:
    """
    Блок численного дифференцирования с фильтрацией.

    Вычисляет вектор скорости по последовательным измерениям координат
    методом конечных разностей назад (backward difference).
    Результат дополнительно сглаживается ФНЧ первого порядка (EMA)
    для подавления шума квантования энкодеров.

    Parameters
    ----------
    dt : float
        Период дискретизации (с).
    cutoff_hz : float | None
        Частота среза ФНЧ для сглаживания скорости (Гц).
        Если ``None`` — фильтрация отключена (сырая производная).
    """

    def __init__(self, dt: float, cutoff_hz: float | None = None) -> None:
        self._dt = float(dt)
        self._prev_positions: NDArray[np.float64] | None = None
        self._filtered_velocity: NDArray[np.float64] | None = None

        # Коэффициент EMA-фильтра: alpha = dt / (tau + dt)
        if cutoff_hz is not None and cutoff_hz > 0.0:
            tau = 1.0 / (2.0 * np.pi * cutoff_hz)
            self._alpha = self._dt / (tau + self._dt)
        else:
            self._alpha = 1.0  # без фильтрации

    # ── Основной метод ────────────────────────────────────────────────────

    def calculate_velocity(self, positions: np.ndarray) -> np.ndarray:
        """
        Вычислить скорость по текущему вектору координат.

        Parameters
        ----------
        positions : State
            Координаты ``(x, θ₁, θ₂)`` на текущем шаге.

        Returns
        -------
        State
            Скорости ``(ẋ, θ̇₁, θ̇₂)``.
        """
        if self._prev_positions is None:
            self._prev_positions = positions.copy()
            return np.zeros_like(positions)

        # Сырая производная (backward difference)
        raw_vel = (positions - self._prev_positions) / self._dt

        # EMA-сглаживание
        if self._filtered_velocity is None:
            self._filtered_velocity = raw_vel.copy()
        else:
            self._filtered_velocity = (
                (1.0 - self._alpha) * self._filtered_velocity
                + self._alpha * raw_vel
            )

        self._prev_positions = positions.copy()
        return self._filtered_velocity

    # ── Сброс ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Сбросить внутреннюю историю (вызывать перед каждым эпизодом)."""
        self._prev_positions = None
        self._filtered_velocity = None


# ═══════════════════════════════════════════════════════════════════════════
# SignalFilter
# ═══════════════════════════════════════════════════════════════════════════

class SignalFilter:
    """
    Блок экспоненциального сглаживания (ФНЧ первого порядка).

    Реализует фильтр :math:`y_k = (1-α)·y_{k-1} + α·u_k`
    с коэффициентом :math:`α = dt / (τ + dt)`, где
    :math:`τ = 1 / (2π·f_{cut})`.

    Parameters
    ----------
    cutoff_hz : float
        Частота среза фильтра (Гц).
    dt : float
        Период дискретизации (с).
    """

    def __init__(self, cutoff_hz: float, dt: float) -> None:
        tau = 1.0 / (2.0 * np.pi * cutoff_hz)
        self._alpha: float = dt / (tau + dt)
        self._filtered: NDArray[np.float64] | None = None

    # ── Основной метод ────────────────────────────────────────────────────

    def filter_signal(self, measurement: np.ndarray) -> np.ndarray:
        """
        Пропустить измерение через ФНЧ.

        Parameters
        ----------
        measurement : MeasuredState
            Входной зашумлённый вектор состояния.

        Returns
        -------
        MeasuredState
            Сглаженный вектор состояния.
        """

        if self._filtered is None:
            self._filtered = measurement.copy()
        else:
            self._filtered = (
                (1.0 - self._alpha) * self._filtered
                + self._alpha * measurement
            )

        return self._filtered

    # ── Сброс ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Сбросить внутреннюю память фильтра."""
        self._filtered = None


# ═══════════════════════════════════════════════════════════════════════════
# Controller (Abstract)
# ═══════════════════════════════════════════════════════════════════════════

class Controller(ABC):
    """
    Абстрактное устройство управления (регулятор).

    Реализует **шаблонный метод** (Template Method) :meth:`compute_control`,
    который задаёт жёсткий конвейер обработки сигнала,
    общий для любых законов управления:

    1. Приём ``measured_state``
    2. Вычисление скоростей (дифференцирование) при отсутствии датчиков
    3. Фильтрация всего вектора состояния
    4. Вызов абстрактного :meth:`get_action` (закон управления)
    5. Ограничение по насыщению привода
    6. Сохранение и возврат ``F_ideal``
    """

    def __init__(self, config: ControllerConfig) -> None:
        """
        Parameters
        ----------
        config : dict | ControllerConfig
            Конфигурация. ``ControllerConfig`` — типизированный датакласс
            из ``datatypes.py``. ``dict`` — плоский словарь с теми же
            ключами для обратной совместимости.
        """
        # ── Извлечение параметров ─────────────────────────────────────

        _dt = config.dt
        _max_force = config.max_force
        _has_vel = config.has_velocity_sensors
        _diff_cutoff = config.differentiator_cutoff_hz
        _filter_cutoff = config.filter_cutoff_hz

        self.name:str
        self._dt = _dt
        self._max_force = _max_force
        self._has_velocity_sensors = _has_vel

        # Компоненты обработки сигналов
        self._differentiator = Differentiator(
            dt=self._dt,
            cutoff_hz=_diff_cutoff if _diff_cutoff is not None else None,
        )
        self._signal_filter = SignalFilter(
            cutoff_hz=_filter_cutoff, dt=self._dt
        )

        # Память
        self._last_control_action: float = 0.0
        self._motor_inertia: MotorInertia | None = None
        # Блоки сглаживания для управляющего сигнала
        # action_filter — сглаживание на этапе ДО ограничения (для подавления импульсов)
        # action_smooth — сглаживание итогового (обрязанного) сигнала перед выдачей
        _action_filter_cutoff = getattr(config, "action_filter_cutoff_hz", None)
        _action_smooth_cutoff = getattr(config, "action_smoothing_cutoff_hz", None)
        if _action_filter_cutoff is not None and _action_filter_cutoff > 0.0:
            tau_af = 1.0 / (2.0 * np.pi * float(_action_filter_cutoff))
            self._alpha_action_filter = float(self._dt) / (tau_af + float(self._dt))
        else:
            self._alpha_action_filter = 1.0
        if _action_smooth_cutoff is not None and _action_smooth_cutoff > 0.0:
            tau_as = 1.0 / (2.0 * np.pi * float(_action_smooth_cutoff))
            self._alpha_action_smooth = float(self._dt) / (tau_as + float(self._dt))
        else:
            self._alpha_action_smooth = 1.0
        self._action_filtered: float | None = None
        self._action_smoothed: float | None = None

    def set_motor_inertia(self, time_constant: float) -> None:
        """Установить модель инерционности двигателя."""
        self._motor_inertia = MotorInertia(time_constant)

    # ── Свойства ──────────────────────────────────────────────────────────

    @property
    def last_control_action(self) -> float:
        """Последнее вычисленное значение силы (Н)."""
        return self._last_control_action

    @property
    def differentiator(self) -> Differentiator:
        """Блок численного дифференцирования."""
        return self._differentiator

    @property
    def signal_filter(self) -> SignalFilter:
        """Блок фильтрации сигнала."""
        return self._signal_filter

    # ── Шаблонный метод ───────────────────────────────────────────────────
    def compute_control(self, measured_state: np.ndarray, target_state:np.ndarray) -> float:
        """
        Основной рабочий метод (Template Method).

        Pipeline:
        1. Извлечь координаты из measured_state.
        2. Вычесть из measured_state, target_state (error).
        3. Если датчиков скоростей нет — вычислить скорости через
           ``differentiator.calculate_velocity()``, иначе взять
           значения из ``measured_state``.
        4. Собрать полный ``MeasuredState`` и пропустить через
           ``signal_filter.filter_signal()``.
        5. Вызвать абстрактный :meth:`get_action(e_clean)`.
        6. Ограничить силу диапазоном ``[-max_force, +max_force]``.
        7. Сохранить в ``last_control_action`` и вернуть.

        Parameters
        ----------
        measured_state : MeasuredState
            Зашумлённый и/или квантованный вектор состояния с датчиков.

        Returns
        -------
        float
            Идеальная управляющая сила ``F_ideal`` (Н).
        """

        # ── 2. Скорости ────────────────────────────────────────────────
        if self._has_velocity_sensors:
            velocities = measured_state[3:]
        else:
            velocities = self._differentiator.calculate_velocity(measured_state[:3])

        # ── 3. Фильтрация ──────────────────────────────────────────────
        full = np.concat([measured_state[:3], velocities])
        s_clean = self._signal_filter.filter_signal(full)

        # ── 4. Закон управления (абстрактный) ──────────────────────────
        F_raw = self.get_action(s_clean, target_state)

        # ── 5. Сглаживание сформированного сигнала до ограничения (action_filter)
        if self._action_filtered is None:
            self._action_filtered = float(F_raw)
        else:
            self._action_filtered = (
                (1.0 - self._alpha_action_filter) * self._action_filtered
                + self._alpha_action_filter * float(F_raw)
            )

        # ── 6. Насыщение (clipping)
        max_f = self._max_force
        F = self._action_filtered
        if F > max_f:
            F_clipped = max_f
        elif F < -max_f:
            F_clipped = -max_f
        else:
            F_clipped = F
        # ── 7. Сглаживание итогового (обрезанного) сигнала перед выдачей (action_smooth)
        if self._action_smoothed is None:
            self._action_smoothed = F_clipped
        else:
            self._action_smoothed = (
                (1.0 - self._alpha_action_smooth) * self._action_smoothed
                + self._alpha_action_smooth * F_clipped
            )

        # ── 8. Модель инерционности мотора (если задана) применяется к сглаженному сигналу
        output_F = self._action_smoothed
        if self._motor_inertia is not None:
            output_F = self._motor_inertia.update(output_F, self._dt)

        # ── 9. Сохранение и возврат
        self._last_control_action = float(output_F)
        return self._last_control_action

    # ── Абстрактный метод (закон управления) ──────────────────────────────

    @abstractmethod
    def get_action(self, s_clean: np.ndarray, target_state:np.ndarray) -> float:
        """
        Абстрактный метод вычисления управляющего воздействия.

        Переопределяется в классах-наследниках для реализации конкретного
        закона управления (ПИД, LQR, нейросеть и т.д.).

        Parameters
        ----------
        s_clean : MeasuredState
            Отфильтрованный вектор состояния ``(x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂)``.

        Returns
        -------
        float
            Идеальная сила (Н) **до** насыщения.
        """
        ...

    # ── Сброс ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """
        Сбросить внутреннюю память фильтра и дифференциатора.

        Вызывать в начале каждого нового эпизода, чтобы переходные
        процессы предыдущего запуска не влияли на старт.
        """
        self._differentiator.reset()
        self._signal_filter.reset()
        if self._motor_inertia is not None:
            self._motor_inertia.reset()
        self._last_control_action = 0.0
