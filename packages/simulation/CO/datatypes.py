from __future__ import annotations

from dataclasses import dataclass, field
import random
import numpy as np


@dataclass
class NoiseForce:
    """
    Мгновенное значение силы внешнего возмущения (белый шум).

    Parameters
    ----------
    mean : float
        Математическое ожидание внешнего возмущения (Н).
    std : float
        СКО (среднеквадратичное отклонение) внешнего возмущения (Н).

    Examples
    --------
    >>> noise = NoiseForce(mean=0.05, std=0.02)
    >>> force = noise.get_force()
    >>> isinstance(force, float)
    True
    """

    mean: float = 0.0
    std: float = 0.0

    def get_force(self) -> float:
        """
        Сгенерировать случайную силу по нормальному распределению.

        Returns
        -------
        float
            Случайное значение силы (Н) согласно ``mean`` и ``std``.
            Если ``std == 0``, возвращает ``mean`` без генерации.
        """
        if self.std == 0.0:
            return self.mean
        return random.gauss(self.mean, self.std)


# ═══════════════════════════════════════════════════════════════════════════
# Конфигурации
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PlantConfig:
    """
    Конфигурация физических параметров объекта управления (тележка + маятник).

    Параметры модели: тележка массы :math:`M` с двухзвенным маятником
    (массы :math:`m_1,m_2`, длины :math:`l_1,l_2`) в поле тяжести :math:`g`.

    Расстояния до центров масс (:math:`L_1,L_2`) и моменты инерции
    (:math:`J_1,J_2`) рассчитываются автоматически в предположении,
    что звенья — однородные стержни:
        :math:`L_i = l_i / 2`,
        :math:`J_i = \\frac{1}{12} m_i l_i^2`.

    Parameters
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
        Флаг учёта люфта редуктора (TODO: может быть удалён).
    backlash_alpha : float
        Ширина зазора редуктора (м).
    backlash_m_mot : float
        Приведённая масса ротора двигателя (кг).
    init_q : np.ndarray
        Начальные обобщённые координаты ``(x, θ₁, θ₂)``.
    init_dq : np.ndarray
        Начальные обобщённые скорости ``(ẋ, θ̇₁, θ̇₂)``.
    dt : float
        Шаг интегрирования физики (с).

    Notes
    -----
    Optimization potential:
        - ``copy()`` создаёт полную копию на каждой итерации обучения;
          для горячего цикла можно переиспользовать экземпляр через ``reset()``
          объекта ``ObjectOfControl``.
        - ``to_dict()`` вызывается однократно при создании ``ObjectOfControl``,
          не является узким местом.
    """

    M: float = 1.0
    m1: float = 0.3
    m2: float = 0.0
    l1: float = 1.0
    l2: float = 0.0
    g: float = 9.81

    b_c: float = 0.0
    b_1: float = 0.0
    b_2: float = 0.0

    single_pendulum_mode: bool = True
    backslash_mode: bool = False
    backlash_alpha: float = 0.0
    backlash_m_mot: float = 0.0

    init_q: np.ndarray = field(default_factory=lambda: np.array([0.0, np.pi, 0.0]))
    init_dq: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))

    dt: float = 0.0005

    # ── Вычисляемые поля (заполняются в __post_init__) ────────────────
    L1: float = field(init=False)
    L2: float = field(init=False)
    J1: float = field(init=False)
    J2: float = field(init=False)

    def __post_init__(self) -> None:
        """Рассчитать L1, L2, J1, J2 на основе m1, m2, l1, l2."""
        # Для однородного стержня:
        #   L_i = l_i / 2  — расстояние от оси до центра масс
        #   J_i = (1/12) * m_i * l_i²  — момент инерции относительно ЦМ
        object.__setattr__(self, "L1", self.l1 / 2.0)
        object.__setattr__(self, "L2", self.l2 / 2.0)
        object.__setattr__(self, "J1", (1.0 / 12.0) * self.m1 * self.l1**2)
        object.__setattr__(self, "J2", (1.0 / 12.0) * self.m2 * self.l2**2)

    def to_dict(self) -> dict:
        """
        Преобразовать в плоский словарь для ``ObjectOfControl.__init__``.

        Returns
        -------
        dict
            Словарь с физическими параметрами, совместимый с ``ObjectOfControl``.
        """
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
            "dt": self.dt
        }

    def copy(self) -> PlantConfig:
        """
        Создать глубокую копию конфигурации.

        Returns
        -------
        PlantConfig
            Независимая копия со своими массивами ``init_q`` / ``init_dq``.
        """
        return PlantConfig(
            M=self.M,
            m1=self.m1,
            m2=self.m2,
            l1=self.l1,
            l2=self.l2,
            g=self.g,
            b_c=self.b_c,
            b_1=self.b_1,
            b_2=self.b_2,
            single_pendulum_mode=self.single_pendulum_mode,
            backslash_mode=self.backslash_mode,
            backlash_alpha=self.backlash_alpha,
            backlash_m_mot=self.backlash_m_mot,
            init_q=self.init_q.copy(),
            init_dq=self.init_dq.copy(),
            dt=self.dt,
        )


@dataclass
class SensorConfig:
    """
    Конфигурация датчиков и шумов измерительной подсистемы.

    Parameters
    ----------
    encoder_resolution_1 : int
        Разрядность энкодера первого звена (импульсов на оборот).
    encoder_resolution_2 : int
        Разрядность энкодера второго звена.
    cart_sensor_resolution : float
        Дискретность датчика положения каретки (м).
    noise_std_q : list[float] | tuple[float, float, float]
        СКО белого шума для координат ``[x, θ₁, θ₂]``.
    noise_std_dq : list[float] | tuple[float, float, float]
        СКО белого шума для скоростей ``[ẋ, θ̇₁, θ̇₂]``.
    seed : int | None
        Seed для генератора шума (воспроизводимость).
        ``None`` — случайные шумы при каждом запуске.

    Notes
    -----
    SensorBlock предвычисляет пул шумов размера ``pool_size=2_000_000``
    при инициализации, что ускоряет симуляцию (не нужно генерировать
    случайные числа на каждом такте).

    Optimization potential:
        - ``pool_size`` мог бы быть параметром конфигурации
          (сейчас захардкожен в ``SensorBlock``).
        - При ``seed=None`` каждый запуск даёт разные шумы,
          что полезно для обучения, но мешает отладке.
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
        """
        Преобразовать в плоский словарь для ``SensorBlock.__init__``.

        Returns
        -------
        dict
            Словарь с параметрами датчиков.
        """
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

    Определяет параметры обработки сигнала в ``Controller``:
    такт управления, ограничение силы, настройки фильтров.

    Parameters
    ----------
    dt : float
        Такт УУ (с), по умолч. 0.005 (200 Гц).
    max_force : float
        Максимальная сила мотора (Н), по умолч. 30.0.
    has_velocity_sensors : bool
        ``True``, если скорости измеряются аппаратно
        (иначе вычисляются дифференциатором).
    differentiator_cutoff_hz : float | None
        Частота среза ФНЧ дифференциатора (``None`` — без фильтрации).
    filter_cutoff_hz : float
        Частота среза ФНЧ сигнала (Гц).

    Notes
    -----
    Ранее существовавшие поля ``action_filter_cutoff_hz`` и
    ``action_smoothing_cutoff_hz`` удалены как избыточные
    (см. ``compute_control`` в ``controller.py``).

    Поле ``gains`` удалено — теперь коэффициенты PID передаются
    напрямую в конструктор ``PIDController`` как ``np.ndarray``.
    """

    dt: float = 0.005
    max_force: float = 30.0
    has_velocity_sensors: bool = False
    differentiator_cutoff_hz: float | None = None
    filter_cutoff_hz: float = 50.0

    def to_dict(self) -> dict:
        """
        Преобразовать в плоский словарь для ``Controller.__init__``.

        Returns
        -------
        dict
            Словарь с параметрами регулятора.
        """
        return {
            "dt": self.dt,
            "max_force": self.max_force,
            "has_velocity_sensors": self.has_velocity_sensors,
            "differentiator_cutoff_hz": self.differentiator_cutoff_hz,
            "filter_cutoff_hz": self.filter_cutoff_hz,
        }
