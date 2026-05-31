import numpy as np
from numpy.typing import NDArray

from .datatypes import NoiseForce, State, StateDot


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
            # Контакт по положительной стороне — сила передаётся
            overtravel = self._gap_pos - self._half_gap
            self._gap_pos = self._half_gap
            # Упругий контакт: избыток пропорционален силе
            # (жёсткость = m_mot / dt^2, приведённая)
            F_real = F_ideal
        elif self._gap_pos < -self._half_gap:
            # Контакт по отрицательной стороне
            overtravel = -self._half_gap - self._gap_pos
            self._gap_pos = -self._half_gap
            F_real = F_ideal
        else:
            # Внутри зазора — усилия нет
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

    def __init__(self, config: dict) -> None:
        """
        Parameters
        ----------
        config : dict
            Словарь конфигурации со следующими ключами:

            **Физические параметры**
            - ``M`` : ``float`` — масса тележки (кг).
            - ``m1`` : ``float`` — масса первого звена (кг).
            - ``m2`` : ``float`` — масса второго звена (кг).
            - ``l1`` : ``float`` — длина первого звена (м).
            - ``l2`` : ``float`` — длина второго звена (м).
            - ``L1`` : ``float`` — расстояние до ЦМ первого звена (м).
            - ``L2`` : ``float`` — расстояние до ЦМ второго звена (м).
            - ``J1`` : ``float`` — момент инерции первого звена (кг·м²).
            - ``J2`` : ``float`` — момент инерции второго звена (кг·м²).
            - ``g`` : ``float``, optional — ускорение свободного падения
              (по умолчанию 9.81 м/с²).

            **Коэффициенты демпфирования** (опционально, по умолч. 0)
            - ``b_c`` : ``float`` — вязкое трение тележки.
            - ``b_1`` : ``float`` — вязкое трение в шарнире первого звена.
            - ``b_2`` : ``float`` — вязкое трение в шарнире второго звена.

            **Режимы**
            - ``single_pendulum_mode`` : ``bool`` — блокировка второй
              степени свободы.
            - ``backslash_mode`` : ``bool`` — учитывать люфт редуктора.

            **Параметры люфта** (если ``backslash_mode=True``)
            - ``backlash_alpha`` : ``float`` — ширина зазора (м).
            - ``backlash_m_mot`` : ``float`` — масса ротора (кг).

            **Начальное состояние** (опционально)
            - ``init_q`` : ``list[float]`` — ``[x, θ₁, θ₂]`` (рад, м).
            - ``init_dq`` : ``list[float]`` — ``[ẋ, θ̇₁, θ̇₂]`` (рад/с, м/с).
        """
        # ── Физические константы ──────────────────────────────────────
        self._M: float = float(config["M"])
        self._m1: float = float(config["m1"])
        self._m2: float = float(config["m2"])
        self._l1: float = float(config["l1"])
        self._l2: float = float(config["l2"])
        self._L1: float = float(config["L1"])
        self._L2: float = float(config["L2"])
        self._J1: float = float(config["J1"])
        self._J2: float = float(config["J2"])
        self._g: float = float(config.get("g", -9.81))

        # ── Демпфирование (по умолчанию 0 — пренебрегаем трением) ─────
        self._b_c: float = float(config.get("b_c", 0.0))
        self._b_1: float = float(config.get("b_1", 0.0))
        self._b_2: float = float(config.get("b_2", 0.0))

        # ── Режимы ────────────────────────────────────────────────────
        self._single_mode: bool = bool(config.get("single_pendulum_mode", False))
        self._backslash_mode: bool = bool(config.get("backslash_mode", False))

        # ── Вектор состояния ──────────────────────────────────────────
        init_q = config.get("init_q", [0.0, 0.0, 0.0])
        init_dq = config.get("init_dq", [0.0, 0.0, 0.0])
        self._q: NDArray[np.float64] = np.asarray(init_q, dtype=np.float64)
        self._dq: NDArray[np.float64] = np.asarray(init_dq, dtype=np.float64)

        # ── Модель люфта ──────────────────────────────────────────────
        if self._backslash_mode:
            alpha = float(config.get("backlash_alpha", 0.0))
            m_mot = float(config.get("backlash_m_mot", 0.0))
            self._backlash = BacklashModel(alpha, m_mot)
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

    def _compute_lagrange_equations(
        self, F_total: float
    ) -> NDArray[np.float64]:
        """
        Разрешить матричное уравнение Лагранжа 2-го рода.

        Решает систему:
        .. math::
            M(θ_1, θ_2) \\cdot \\ddot{q} + C(q, \\dot{q})
            + G(θ_1, θ_2) = Q

        Parameters
        ----------
        F_total : float
            Суммарная внешняя сила, приложенная к тележке (Н).

        Returns
        -------
        NDArray[np.float64]
            Вектор обобщённых ускорений ``[ẍ, θ̈₁, θ̈₂]``.
        """
        # В одно-звенном режиме принудительно зануляем второе звено
        if self._single_mode:
            self._q[2] = 0.0
            self._dq[2] = 0.0

        # Распаковка состояния
        x, th1, th2 = self._q
        dx, dth1, dth2 = self._dq

        # Предвычисление тригонометрических функций
        c1 = np.cos(th1)
        s1 = np.sin(th1)
        c12 = np.cos(th1 + th2)  # cos(θ₁ + θ₂)
        s12 = np.sin(th1 + th2)  # sin(θ₁ + θ₂)
        c2 = np.cos(th2)
        s2 = np.sin(th2)

        # ── Матрица масс M (3×3) ──────────────────────────────────────
        A = self._m1 * self._L1 + self._m2 * self._l1  # m1·L1 + m2·l1
        B = self._m2 * self._L2                         # m2·L2

        M11 = self._M + self._m1 + self._m2
        M12 = A * c1 + B * c12
        M13 = B * c12

        # M21 = M12 (симметрия)
        M22 = (
            self._J1
            + self._m1 * self._L1 ** 2
            + self._J2
            + self._m2 * (self._l1 ** 2 + self._L2 ** 2 + 2.0 * self._l1 * self._L2 * c2)
        )
        M23 = self._J2 + self._m2 * self._L2 ** 2 + self._m2 * self._l1 * self._L2 * c2

        # M31 = M13, M32 = M23
        M33 = self._J2 + self._m2 * self._L2 ** 2

        # ── Матрица кориолисовых / центробежных членов C (3×3) ────────
        # C11 = C21 = C31 = 0
        K = A * dth1 * s1 + B * (dth1 + dth2) * s12  # общая часть для C12, C13
        C12 = -K
        C13 = -B * (dth1 + dth2) * s12

        C21 = 0.0
        C22 = -self._m2 * self._l1 * self._L2 * s2 * dth2
        C23 = -self._m2 * self._l1 * self._L2 * s2 * (dth1 + dth2)

        C31 = 0.0
        C32 = self._m2 * self._l1 * self._L2 * s2 * dth1
        C33 = 0.0

        # ── Вектор гравитации G ───────────────────────────────────────
        # G1 = 0 (тележка на горизонтальном треке)
        # G2 = -(m₁·L₁ + m₂·l₁)·g·sinθ₁ - m₂·L₂·g·sin(θ₁+θ₂)
        # G3 = -m₂·L₂·g·sin(θ₁+θ₂)
        G1 = 0.0
        G2 = -A * self._g * s1 - B * self._g * s12
        G3 = -B * self._g * s12

        # ── Вектор обобщённых сил Q ───────────────────────────────────
        # Q = [F_total - b_c·ẋ,  -b₁·θ̇₁,  -b₂·θ̇₂]
        Q1 = F_total - self._b_c * dx
        Q2 = -self._b_1 * dth1
        Q3 = -self._b_2 * dth2

        # ── Перенос неинерционных членов в правую часть ───────────────
        # M · ddq = Q - C · dq - G
        rhs1 = (
            Q1
            - C12 * dth1 - C13 * dth2
        )
        rhs2 = (
            Q2
            - C21 * dx - C22 * dth1 - C23 * dth2
            - G2
        )
        rhs3 = (
            Q3
            - C31 * dx - C32 * dth1 - C33 * dth2
            - G3
        )

        # ── Сборка матрицы и решение ───────────────────────────────────
        M = np.array(
            [
                [M11, M12, M13],
                [M12, M22, M23],
                [M13, M23, M33],
            ],
            dtype=np.float64,
        )
        rhs = np.array([rhs1, rhs2, rhs3], dtype=np.float64)

        # Если матрица вырождена (m₂=J₂=0 → M33=0), решаем 2×2 подсистему
        if self._single_mode and M33 == 0.0:
            # [M11  M12] [ẍ]   = [rhs1]
            # [M12  M22] [θ̈₁]   [rhs2]
            det = M11 * M22 - M12 * M12
            if abs(det) > 1e-15:
                ddx = (rhs1 * M22 - M12 * rhs2) / det
                dth1_2 = (M11 * rhs2 - M12 * rhs1) / det
                ddq = np.array([ddx, dth1_2, 0.0], dtype=np.float64)
            else:
                ddq = np.zeros(3, dtype=np.float64)
        else:
            ddq: NDArray[np.float64] = np.linalg.solve(M, rhs)  # type: ignore[assignment]

        # В одно-звенном режиме дополнительно блокируем ускорение второго звена
        if self._single_mode:
            ddq[2] = 0.0

        return ddq

    # ──────────────────────────────────────────────────────────────────────
    # Шаг интегрирования
    # ──────────────────────────────────────────────────────────────────────

    def _state_dot(
        self, F_total: float
    ) -> NDArray[np.float64]:
        """
        Вычислить производную полного вектора состояния
        ``state = [q, dq]`` → ``state_dot = [dq, ddq]``.

        Parameters
        ----------
        F_total : float
            Суммарная сила на тележке (Н).

        Returns
        -------
        NDArray[np.float64]
            Вектор ``[ẋ, θ̇₁, θ̇₂, ẍ, θ̈₁, θ̈₂]``.
        """
        ddq = self._compute_lagrange_equations(F_total)
        return np.concatenate([self._dq, ddq])

    def _rk4_step(
        self, F_total: float, dt: float
    ) -> None:
        """
        Интегрирование состояния на шаг ``dt`` методом Рунге — Кутты 4-го порядка.

        Parameters
        ----------
        F_total : float
            Суммарная сила на тележке (Н) — **постоянна на шаге**.
        dt : float
            Шаг интегрирования (с).
        """
        # Функция-обёртка для расчёта правой части
        def f(state: NDArray[np.float64]) -> NDArray[np.float64]:
            # Распаковать состояние
            x_, th1_, th2_, dx_, dth1_, dth2_ = state
            # Восстановить внутренние векторы
            self._q[:] = [x_, th1_, th2_]
            self._dq[:] = [dx_, dth1_, dth2_]
            return self._state_dot(F_total)

        # Текущее состояние
        s = np.concatenate([self._q, self._dq])

        # Коэффициенты RK4
        k1 = f(s)
        k2 = f(s + 0.5 * dt * k1)
        k3 = f(s + 0.5 * dt * k2)
        k4 = f(s + dt * k3)

        s_next = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Обновить внутренние векторы
        self._q[:] = s_next[:3]
        self._dq[:] = s_next[3:]

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
        # Шаг 1 — учёт люфта
        if self._backslash_mode and self._backlash is not None:
            F_real = self._backlash.update(F_ideal, self._dq[0], dt_physics)
        else:
            F_real = F_ideal

        # Шаг 2 — суммарная сила
        F_total = F_real + F_noise.value

        # Шаг 3 — интегрирование
        self._rk4_step(F_total, dt_physics)

    def compute_lagrange_equations(
        self, F_total: float
    ) -> NDArray[np.float64]:
        """
        Публичная обёртка над вычислительным ядром.

        Parameters
        ----------
        F_total : float
            Суммарная внешняя сила на тележке (Н).

        Returns
        -------
        NDArray[np.float64]
            Вектор ускорений ``[ẍ, θ̈₁, θ̈₂]``.
        """
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
