import numpy as np
from numba import njit
from numpy.typing import NDArray

from .datatypes import NoiseForce, PlantConfig, State, StateDot


# ═══════════════════════════════════════════════════════════════════════════
# BacklashModel
# ═══════════════════════════════════════════════════════════════════════════

class BacklashModel:
    """
    Модель люфта (зазора) механического редуктора привода.

    Пока мотор находится внутри зазора шириной ``alpha``, усилие на тележку
    не передаётся (``F_real = 0``). Как только зазор полностью выбран с одной
    из сторон, усилие передаётся полностью (``F_real = F_ideal``).

    Parameters
    ----------
    alpha : float
        Ширина зазора редуктора (в линейном перемещении, м).
    m_mot : float
        Приведённая масса ротора двигателя (кг).
    """

    def __init__(self, alpha: float, m_mot: float) -> None:
        self._alpha = float(alpha)
        self._m_mot = float(m_mot)
        self._half_gap = self._alpha / 2.0

        # Внутреннее состояние — текущее положение внутри зазора.
        # Диапазон: [-half_gap, +half_gap].
        self._gap_pos: float = 0.0

    # ── Свойства ──────────────────────────────────────────────────────────

    @property
    def alpha(self) -> float:
        """Ширина зазора редуктора (м)."""
        return self._alpha

    @property
    def gap_position(self) -> float:
        """
        Текущее относительное положение внутри зазора (м).
        :math:`[-\\alpha/2, +\\alpha/2]`.
        """
        return self._gap_pos

    @property
    def in_contact(self) -> bool:
        """
        ``True``, если зазор полностью выбран (контакт есть).
        """
        return abs(self._gap_pos) >= self._half_gap

    # ── Основной метод ────────────────────────────────────────────────────

    def update(
        self, F_ideal: float, cart_velocity: float, dt: float
    ) -> float:
        """
        Обновить состояние люфта и вернуть реальное усилие на тележку.

        Алгоритм:

        1. Вычислить ускорение ротора относительно тележки
           :math:`a_{rel} = F_{ideal} / m_{mot}`.
        2. Обновить положение внутри зазора:
           :math:`gap + = a_{rel} * dt`.
        3. Если положение выходит за пределы :math:`[-alpha/2, +alpha/2]`,
           избыток считается силой контакта, передаваемой на тележку.
           Положение фиксируется на границе.

        Parameters
        ----------
        F_ideal : float
            Идеальное управляющее усилие (Н).
        cart_velocity : float
            Текущая скорость тележки (м/с) — пока не используется
            в простейшей модели (задел для вязкого трения в зазоре).
        dt : float
            Шаг интегрирования (с).

        Returns
        -------
        float
            Реальная сила на тележке ``F_real`` (Н).
        """
        # 1. Ускорение ротора относительно тележки внутри зазора
        a_rel = F_ideal / self._m_mot

        # 2. Обновление положения
        self._gap_pos += a_rel * dt

        # 3. Проверка контакта
        if self._gap_pos > self._half_gap:
            self._gap_pos = self._half_gap
            F_real = F_ideal
        elif self._gap_pos < -self._half_gap:
            self._gap_pos = -self._half_gap
            F_real = F_ideal
        else:
            F_real = 0.0

        return F_real


# ═══════════════════════════════════════════════════════════════════════════
# ObjectOfControl
# ═══════════════════════════════════════════════════════════════════════════

class ObjectOfControl:
    """
    Математическая модель физической части системы — тележка
    с многозвенным (одно- или двухзвенным) маятником.

    Выполняет непрерывное интегрирование уравнений движения
    методом Рунге — Кутты 4-го порядка на физическом микрошаге
    ``dt_physics`` и инкапсулирует нелинейность привода (люфт).
    """

    # ──────────────────────────────────────────────────────────────────────
    # Конструктор
    # ──────────────────────────────────────────────────────────────────────

    def __init__(self, config: PlantConfig) -> None:
        """
        Parameters
        ----------
        config : PlantConfig
            Типизированная конфигурация физических параметров ОУ.
        """
        # ── Физические константы ──────────────────────────────────────
        self._M: float = config.M
        self._m1: float = config.m1
        self._m2: float = config.m2
        self._l1: float = config.l1
        self._l2: float = config.l2
        self._L1: float = config.L1
        self._L2: float = config.L2
        self._J1: float = config.J1
        self._J2: float = config.J2
        self._g: float = config.g

        # ── Демпфирование ─────────────────────────────────────────────
        self._b_c: float = config.b_c
        self._b_1: float = config.b_1
        self._b_2: float = config.b_2

        # ── Режимы ────────────────────────────────────────────────────
        self._single_mode: bool = config.single_pendulum_mode
        self._backslash_mode: bool = config.backslash_mode

        # ── Вектор состояния ──────────────────────────────────────────
        self._q: NDArray[np.float64] = np.asarray(
            list(config.init_q), dtype=np.float64,
        )
        self._dq: NDArray[np.float64] = np.asarray(
            list(config.init_dq), dtype=np.float64,
        )

        # ── Модель люфта ──────────────────────────────────────────────
        if self._backslash_mode:
            self._backlash = BacklashModel(config.backlash_alpha, config.backlash_m_mot)
        else:
            self._backlash = None

    # ──────────────────────────────────────────────────────────────────────
    # Свойства
    # ──────────────────────────────────────────────────────────────────────

    @property
    def q(self) -> NDArray[np.float64]:
        """Вектор обобщённых координат ``[x, θ₁, θ₂]``."""
        return self._q.copy()

    @property
    def dq(self) -> NDArray[np.float64]:
        """Вектор обобщённых скоростей ``[ẋ, θ̇₁, θ̇₂]``."""
        return self._dq.copy()

    @property
    def backlash_model(self) -> BacklashModel | None:
        """Объект модели люфта (``None``, если люфт не учитывается)."""
        return self._backlash

    @property
    def single_pendulum_mode(self) -> bool:
        """Флаг блокировки второй степени свободы."""
        return self._single_mode

    @single_pendulum_mode.setter
    def single_pendulum_mode(self, value: bool) -> None:
        self._single_mode = bool(value)

    # ──────────────────────────────────────────────────────────────────────
    # Вычислительное ядро — уравнения Лагранжа
    # ──────────────────────────────────────────────────────────────────────

    def _compute_lagrange_equations(self, F_total: float) -> NDArray[np.float64]:
        if self._single_mode:
            self._q[2] = 0.0
            self._dq[2] = 0.0
        
        params = np.array([
            self._M, self._m1, self._m2, self._l1, self._l2,
            self._L1, self._L2, self._J1, self._J2, self._g,
            self._b_c, self._b_1, self._b_2
        ])
        return _compute_ddq_numba(self._q, self._dq, F_total, params, self._single_mode)

    def _rk4_step(self, F_total: float, dt: float) -> None:
        params = np.array([
            self._M, self._m1, self._m2, self._l1, self._l2,
            self._L1, self._L2, self._J1, self._J2, self._g,
            self._b_c, self._b_1, self._b_2
        ])
        self._q, self._dq = _rk4_step_numba(self._q, self._dq, F_total, dt, params, self._single_mode)

    # ──────────────────────────────────────────────────────────────────────
    # Публичный API
    # ──────────────────────────────────────────────────────────────────────

    def update_physics(
        self, F_ideal: float, F_noise: NoiseForce, dt_physics: float
    ) -> None:
        """
        Главный шаг интегрирования физики ОУ.

        Алгоритм:

        1. Если ``backslash_mode`` включён — передать ``F_ideal``
           и скорость тележки в модель люфта для получения ``F_real``.
           Иначе ``F_real = F_ideal``.
        2. Сформировать суммарную силу: ``F_total = F_real + F_noise.value``.
        3. Выполнить один шаг RK4 с силой ``F_total``.

        Parameters
        ----------
        F_ideal : float
            Идеальное управляющее усилие от УУ (Н).
        F_noise : NoiseForce
            Мгновенное значение силы внешнего возмущения.
        dt_physics : float
            Шаг интегрирования физики (с).
        """
        if self._backslash_mode and self._backlash is not None:
            F_real = self._backlash.update(F_ideal, self._dq[0], dt_physics)
        else:
            F_real = F_ideal

        F_total = F_real + F_noise.value
        self._rk4_step(F_total, dt_physics)

    def compute_lagrange_equations(self, F_total: float) -> NDArray[np.float64]:
        return self._compute_lagrange_equations(F_total)

    def get_clean_state(self) -> tuple[State, StateDot]:
        """
        Вернуть абсолютно чистые (неискажённые шумами датчиков)
        координаты и скорости системы.

        Returns
        -------
        tuple[State, StateDot]
            Кортеж ``(q, dq)``.
        """
        return (
            State(x=self._q[0], theta1=self._q[1], theta2=self._q[2]),
            StateDot(
                x_dot=self._dq[0],
                theta1_dot=self._dq[1],
                theta2_dot=self._dq[2],
            ),
        )

@njit(cache=True)
def _compute_ddq_numba(
    q: NDArray[np.float64],
    dq: NDArray[np.float64],
    F_total: float,
    params: NDArray[np.float64],
    single_mode: bool,
) -> NDArray[np.float64]:
    x, th1, th2 = q
    dx, dth1, dth2 = dq
    M, m1, m2, l1, l2, L1, L2, J1, J2, g, b_c, b_1, b_2 = params

    c1, s1 = np.cos(th1), np.sin(th1)
    c12, s12 = np.cos(th1 + th2), np.sin(th1 + th2)
    c2, s2 = np.cos(th2), np.sin(th2)

    A = m1 * L1 + m2 * l1
    B = m2 * L2

    M11 = M + m1 + m2
    M12 = A * c1 + B * c12
    M13 = B * c12
    M22 = J1 + m1 * L1**2 + J2 + m2 * (l1**2 + L2**2 + 2.0 * l1 * L2 * c2)
    M23 = J2 + m2 * L2**2 + m2 * l1 * L2 * c2
    M33 = J2 + m2 * L2**2

    K = A * dth1 * s1 + B * (dth1 + dth2) * s12
    C12, C13 = -K, -B * (dth1 + dth2) * s12
    C22 = -m2 * l1 * L2 * s2 * dth2
    C23 = -m2 * l1 * L2 * s2 * (dth1 + dth2)
    C32 = m2 * l1 * L2 * s2 * dth1

    G2 = -A * g * s1 - B * g * s12
    G3 = -B * g * s12

    rhs1 = (F_total - b_c * dx) - C12 * dth1 - C13 * dth2
    rhs2 = (-b_1 * dth1) - C22 * dth1 - C23 * dth2 - G2
    rhs3 = (-b_2 * dth2) - C32 * dth1 - G3

    if single_mode:
        det = M11 * M22 - M12 * M12
        if abs(det) > 1e-15:
            ddx = (rhs1 * M22 - M12 * rhs2) / det
            dth1_2 = (M11 * rhs2 - M12 * rhs1) / det
            return np.array([ddx, dth1_2, 0.0], dtype=np.float64)
        return np.zeros(3, dtype=np.float64)

    M_mat = np.array([[M11, M12, M13], [M12, M22, M23], [M13, M23, M33]])
    return np.linalg.solve(M_mat, np.array([rhs1, rhs2, rhs3]))

@njit(cache=True)
def _rk4_step_numba(
    q: NDArray[np.float64],
    dq: NDArray[np.float64],
    F_total: float,
    dt: float,
    params: NDArray[np.float64],
    single_mode: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    def get_dot(q_in: NDArray[np.float64], dq_in: NDArray[np.float64]) -> NDArray[np.float64]:
        ddq = _compute_ddq_numba(q_in, dq_in, F_total, params, single_mode)
        res = np.empty(6, dtype=np.float64)
        res[:3] = dq_in
        res[3:] = ddq
        return res

    s = np.empty(6, dtype=np.float64)
    s[:3] = q
    s[3:] = dq
    k1 = get_dot(q, dq)
    k2 = get_dot(q + 0.5 * dt * k1[:3], dq + 0.5 * dt * k1[3:])
    k3 = get_dot(q + 0.5 * dt * k2[:3], dq + 0.5 * dt * k2[3:])
    k4 = get_dot(q + dt * k3[:3], dq + dt * k3[3:])

    s_next = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return s_next[:3], s_next[3:]
