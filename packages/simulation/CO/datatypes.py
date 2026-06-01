from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


# ═══════════════════════════════════════════════════════════════════════════
# Векторы состояния
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class State:
    """
    Вектор обобщённых координат ОУ.

    Attributes
    ----------
    x : float
        Положение тележки (м).
    theta1 : float
        Угол первого звена от вертикали (рад).
    theta2 : float
        Угол второго звена относительно продолжения первого (рад).
    """

    x: float = 0.0
    theta1: float = 0.0
    theta2: float = 0.0


@dataclass
class StateDot:
    """
    Вектор обобщённых скоростей ОУ.

    Attributes
    ----------
    x_dot : float
        Скорость тележки (м/с).
    theta1_dot : float
        Угловая скорость первого звена (рад/с).
    theta2_dot : float
        Угловая скорость второго звена (рад/с).
    """

    x_dot: float = 0.0
    theta1_dot: float = 0.0
    theta2_dot: float = 0.0


@dataclass
class NoiseForce:
    """
    Мгновенное значение силы внешнего возмущения.

    Attributes
    ----------
    value : float
        Сила внешнего возмущения (Н).
    """

    value: float = 0.0


@dataclass
class NoiseConfig:
    """
    Конфигурация внешнего возмущения.

    Attributes
    ----------
    value : float
        Сила внешнего возмущения (Н).
    """

    value: float = 0.0


@dataclass
class MeasuredState:
    """
    Вектор измеренного (зашумлённого и/или квантованного) состояния
    системы, поступающий с датчиков или после дифференцирования.

    Attributes
    ----------
    x : float
        Положение тележки (м).
    theta1 : float
        Угол первого звена (рад).
    theta2 : float
        Угол второго звена (рад).
    x_dot : float
        Скорость тележки (м/с).
    theta1_dot : float
        Угловая скорость первого звена (рад/с).
    theta2_dot : float
        Угловая скорость второго звена (рад/с).
    """

    x: float = 0.0
    theta1: float = 0.0
    theta2: float = 0.0
    x_dot: float = 0.0
    theta1_dot: float = 0.0
    theta2_dot: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Конфигурации
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PlantConfig:
    """
    Конфигурация физических параметров объекта управления (тележка + маятник).

    Attributes
    ----------
    M : float
        Масса тележки (кг).
    m1 : float
        Полная масса первого звена маятника (кг).
    m2 : float
        Полная масса второго звена маятника (кг).
    l1 : float
        Полная геометрическая длина первого звена (м).
    l2 : float
        Полная геометрическая длина второго звена (м).
    L1 : float
        Расстояние от оси вращения тележки до ЦМ первого звена (м).
    L2 : float
        Расстояние от оси промежуточного шарнира до ЦМ второго звена (м).
    J1 : float
        Собственный момент инерции первого звена относительно его ЦМ (кг·м²).
    J2 : float
        Собственный момент инерции второго звена относительно его ЦМ (кг·м²).
    g : float
        Ускорение свободного падения (м/с²).
    b_c : float
        Коэффициент вязкого трения тележки.
    b_1 : float
        Коэффициент вязкого трения в шарнире первого звена.
    b_2 : float
        Коэффициент вязкого трения в шарнире второго звена.
    single_pendulum_mode : bool
        Флаг блокировки второй степени свободы (однозвенный режим).
    backslash_mode : bool
        Флаг учёта люфта редуктора.
    backlash_alpha : float
        Ширина зазора редуктора (м).
    backlash_m_mot : float
        Приведённая масса ротора двигателя (кг).
    init_q : tuple[float, float, float]
        Начальные обобщённые координаты ``(x, θ₁, θ₂)``.
    init_dq : tuple[float, float, float]
        Начальные обобщённые скорости ``(ẋ, θ̇₁, θ̇₂)``.
    """

    M: float = 1.0
    m1: float = 0.3
    m2: float = 0.0
    l1: float = 1.0
    l2: float = 0.0
    L1: float = 0.7
    L2: float = 0.0
    J1: float = 0.02
    J2: float = 0.0
    g: float = 9.81

    b_c: float = 0.0
    b_1: float = 0.0
    b_2: float = 0.0

    single_pendulum_mode: bool = False
    backslash_mode: bool = False
    backlash_alpha: float = 0.0
    backlash_m_mot: float = 0.0

    init_q: tuple[float, float, float] = (0.0, 0.0, 0.0)
    init_dq: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_dict(self) -> dict:
        """Преобразовать в плоский словарь для ``ObjectOfControl.__init__``."""
        return {
            "M": self.M,
            "m1": self.m1,
            "m2": self.m2,
            "l1": self.l1,
            "l2": self.l2,
            "L1": self.L1,
            "L2": self.L2,
            "J1": self.J1,
            "J2": self.J2,
            "g": self.g,
            "b_c": self.b_c,
            "b_1": self.b_1,
            "b_2": self.b_2,
            "single_pendulum_mode": self.single_pendulum_mode,
            "backslash_mode": self.backslash_mode,
            "backlash_alpha": self.backlash_alpha,
            "backlash_m_mot": self.backlash_m_mot,
            "init_q": list(self.init_q),
            "init_dq": list(self.init_dq),
        }


@dataclass
class SensorConfig:
    """
    Конфигурация датчиков и шумов измерительной подсистемы.

    Attributes
    ----------
    encoder_resolution_1 : int
        Разрядность энкодера первого звена (импульсов на оборот).
    encoder_resolution_2 : int
        Разрядность энкодера второго звена.
    cart_sensor_resolution : float
        Дискретность датчика положения каретки (м).
    noise_std_q : NDArray[np.float64]
        СКО белого шума для координат ``[x, θ₁, θ₂]``.
    noise_std_dq : NDArray[np.float64]
        СКО белого шума для скоростей ``[ẋ, θ̇₁, θ̇₂]``.
    noise_std_q : list[float] | tuple[float, float, float]
        СКО белого шума для координат ``[x, θ₁, θ₂]``.
    noise_std_dq : list[float] | tuple[float, float, float]
        СКО белого шума для скоростей ``[ẋ, θ̇₁, θ̇₂]``.
    seed : int | None
        Seed для генератора шума (воспроизводимость).
    """

    encoder_resolution_1: int = 4096
    encoder_resolution_2: int = 4096
    cart_sensor_resolution: float = 0.0001

    noise_std_q: list[float] | tuple[float, float, float] = (
        0.001,
        0.005,
        0.005,
    )
    noise_std_dq: list[float] | tuple[float, float, float] = (
        0.01,
        0.02,
        0.02,
    )
    seed: int | None = None

    def to_dict(self) -> dict:
        """Преобразовать в плоский словарь для ``SensorBlock.__init__``."""
        return {
            "encoder_resolution_1": self.encoder_resolution_1,
            "encoder_resolution_2": self.encoder_resolution_2,
            "cart_sensor_resolution": self.cart_sensor_resolution,
            "noise_std_q": list(self.noise_std_q),
            "noise_std_dq": list(self.noise_std_dq),
            "seed": self.seed,
        }


@dataclass
class ControllerConfig:
    """
    Конфигурация регулятора (Устройства управления).

    Attributes
    ----------
    dt : float
        Такт УУ (с), по умолч. 0.005 (200 Гц).
    max_force : float
        Максимальная сила мотора (Н), по умолч. 30.0.
    has_velocity_sensors : bool
        ``True``, если скорости измеряются аппаратно.
    differentiator_cutoff_hz : float | None
        Частота среза ФНЧ дифференциатора (``None`` — без фильтрации).
    filter_cutoff_hz : float
        Частота среза ФНЧ сигнала (Гц).
    gains : list[float]
        Начальные коэффициенты ``[Kp, Ki, Kd, Kx]`` (для PID).
    """

    dt: float = 0.005
    max_force: float = 30.0
    has_velocity_sensors: bool = False
    differentiator_cutoff_hz: float | None = None
    filter_cutoff_hz: float = 50.0
    gains: list[float] = field(default_factory=lambda: [10.0, 1.0, 2.0, 1.0])

    def to_dict(self) -> dict:
        """Преобразовать в плоский словарь для ``Controller.__init__``."""
        return {
            "dt": self.dt,
            "max_force": self.max_force,
            "has_velocity_sensors": self.has_velocity_sensors,
            "differentiator_cutoff_hz": self.differentiator_cutoff_hz,
            "filter_cutoff_hz": self.filter_cutoff_hz,
            "gains": self.gains,
        }
