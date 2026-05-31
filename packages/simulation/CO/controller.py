from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from .datatypes import MeasuredState


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

    def calculate_velocity(
        self, positions: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Вычислить скорость по текущему вектору координат.

        Parameters
        ----------
        positions : NDArray[np.float64]
            Вектор координат ``[x, θ₁, θ₂]`` на текущем шаге.

        Returns
        -------
        NDArray[np.float64]
            Вектор скоростей ``[ẋ, θ̇₁, θ̇₂]``.
        """
        pos = np.asarray(positions, dtype=np.float64)

        if self._prev_positions is None:
            self._prev_positions = pos.copy()
            return np.zeros(3, dtype=np.float64)

        # Сырая производная (backward difference)
        raw_vel = (pos - self._prev_positions) / self._dt

        # EMA-сглаживание
        if self._filtered_velocity is None:
            self._filtered_velocity = raw_vel.copy()
        else:
            self._filtered_velocity = (
                (1.0 - self._alpha) * self._filtered_velocity
                + self._alpha * raw_vel
            )

        self._prev_positions = pos.copy()
        return self._filtered_velocity.copy()

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
    dim : int
        Размерность фильтруемого вектора (по умолчанию 6).
    """

    def __init__(
        self, cutoff_hz: float, dt: float, dim: int = 6
    ) -> None:
        tau = 1.0 / (2.0 * np.pi * cutoff_hz)
        self._alpha: float = dt / (tau + dt)
        self._dim: int = dim
        self._filtered: NDArray[np.float64] | None = None

    # ── Основной метод ────────────────────────────────────────────────────

    def filter_signal(
        self, measurement: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Пропустить измерение через ФНЧ.

        Parameters
        ----------
        measurement : NDArray[np.float64]
            Входной вектор (зашумлённый).

        Returns
        -------
        NDArray[np.float64]
            Сглаженный вектор.
        """
        meas = np.asarray(measurement, dtype=np.float64)

        if self._filtered is None:
            self._filtered = meas.copy()
        else:
            self._filtered = (
                (1.0 - self._alpha) * self._filtered
                + self._alpha * meas
            )

        return self._filtered.copy()

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

    def __init__(self, config: dict) -> None:
        """
        Parameters
        ----------
        config : dict
            Словарь конфигурации:

            - **dt** : ``float``, optional — такт УУ (с), по умолч. 0.005 (200 Гц).
            - **max_force** : ``float``, optional — макс. сила мотора (Н),
              по умолч. 30.0.
            - **has_velocity_sensors** : ``bool``, optional — ``True``, если
              скорости измеряются аппаратно, а не вычисляются дифференцированием.
            - **differentiator_cutoff_hz** : ``float``, optional — частота среза
              ФНЧ дифференциатора (по умолч. выключен).
            - **filter_cutoff_hz** : ``float``, optional — частота среза ФНЧ
              сигнала (по умолч. 50 Гц).
        """
        # Параметры дискретизации
        self._dt: float = float(config.get("dt", 0.005))
        self._max_force: float = float(config.get("max_force", 30.0))
        self._has_velocity_sensors: bool = bool(
            config.get("has_velocity_sensors", False)
        )

        # Компоненты обработки сигналов
        diff_cutoff = config.get("differentiator_cutoff_hz", None)
        self._differentiator = Differentiator(
            dt=self._dt,
            cutoff_hz=diff_cutoff if diff_cutoff is not None else None,
        )

        filter_cutoff = float(config.get("filter_cutoff_hz", 50.0))
        self._signal_filter = SignalFilter(
            cutoff_hz=filter_cutoff, dt=self._dt, dim=6
        )

        # Память
        self._last_control_action: float = 0.0

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

    def compute_control(self, measured_state: MeasuredState) -> float:
        """
        Основной рабочий метод (Template Method).

        Pipeline:
        1. Извлечь координаты из ``measured_state``.
        2. Если датчиков скоростей нет — вычислить скорости через
           ``differentiator.calculate_velocity()``, иначе взять
           значения из ``measured_state``.
        3. Собрать полный вектор ``[x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂]``
           и пропустить через ``signal_filter.filter_signal()``.
        4. Вызвать абстрактный :meth:`get_action(s_clean)`.
        5. Ограничить силу диапазоном ``[-max_force, +max_force]``.
        6. Сохранить в ``last_control_action`` и вернуть.

        Parameters
        ----------
        measured_state : MeasuredState
            Зашумлённый и/или квантованный вектор состояния с датчиков.

        Returns
        -------
        float
            Идеальная управляющая сила ``F_ideal`` (Н).
        """
        # ── 1. Координаты ──────────────────────────────────────────────
        positions = np.array(
            [measured_state.x, measured_state.theta1, measured_state.theta2],
            dtype=np.float64,
        )

        # ── 2. Скорости ────────────────────────────────────────────────
        if self._has_velocity_sensors:
            velocities = np.array(
                [
                    measured_state.x_dot,
                    measured_state.theta1_dot,
                    measured_state.theta2_dot,
                ],
                dtype=np.float64,
            )
        else:
            velocities = self._differentiator.calculate_velocity(positions)

        # ── 3. Фильтрация ──────────────────────────────────────────────
        full_state = np.concatenate([positions, velocities])
        s_clean = self._signal_filter.filter_signal(full_state)

        # ── 4. Закон управления (абстрактный) ──────────────────────────
        F_raw = self.get_action(s_clean)

        # ── 5. Насыщение ───────────────────────────────────────────────
        F_clipped = float(np.clip(F_raw, -self._max_force, self._max_force))

        # ── 6. Сохранение и возврат ────────────────────────────────────
        self._last_control_action = F_clipped
        return self._last_control_action

    # ── Абстрактный метод (закон управления) ──────────────────────────────

    @abstractmethod
    def get_action(self, s_clean: NDArray[np.float64]) -> float:
        """
        Абстрактный метод вычисления управляющего воздействия.

        Переопределяется в классах-наследниках для реализации конкретного
        закона управления (ПИД, LQR, нейросеть и т.д.).

        Parameters
        ----------
        s_clean : NDArray[np.float64]
            Отфильтрованный вектор состояния
            ``[x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂]``.

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
        self._last_control_action = 0.0
