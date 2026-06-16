from packages.simulation.CO.datatypes import NoiseForce, PlantConfig
import numpy as np

# Optional C++ backend (pybind11). If unavailable (e.g. no built extension), fall back.
# Используем абсолютный импорт, чтобы модуль корректно работал при прямом запуске
# скриптов из репозитория (когда относительные импорты могут не резолвиться).
try:
    from packages.simulation.CO import co_cpp as _co_cpp  # type: ignore
except Exception:  # pragma: no cover
    _co_cpp = None



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
        # Начальное состояние (для reset)
        self._q_init: np.ndarray = config.init_q
        self._dq_init: np.ndarray = config.init_dq

        self._q: np.ndarray = self._q_init
        self._dq: np.ndarray = self._dq_init

        self._dt = config.dt
        # ── Модель люфта ──────────────────────────────────────────────
        if self._backslash_mode:
            self._backlash = BacklashModel(config.backlash_alpha, config.backlash_m_mot)
        else:
            self._backlash = None

        # C++ backend state (gap position)
        self._cpp_backlash_gap_pos: float = 0.0

        # Pre-create C++ objects for reuse (hot-loop optimization)
        self._cpp_noise = None
        self._cpp_params = None
        self._cpp_q = None
        self._cpp_dq = None
        if _co_cpp is not None:
            self._cpp_noise = _co_cpp.NoiseForce(float(0.0), float(0.0))
            self._cpp_params = _co_cpp.PlantParams()
            self._cpp_q = _co_cpp.State3()
            self._cpp_dq = _co_cpp.StateDot3()
            # Fill static params
            self._cpp_params.M = self._M
            self._cpp_params.m1 = self._m1
            self._cpp_params.m2 = self._m2
            self._cpp_params.l1 = self._l1
            self._cpp_params.l2 = self._l2
            self._cpp_params.L1 = self._L1
            self._cpp_params.L2 = self._L2
            self._cpp_params.J1 = self._J1
            self._cpp_params.J2 = self._J2
            self._cpp_params.g = self._g
            self._cpp_params.b_c = self._b_c
            self._cpp_params.b_1 = self._b_1
            self._cpp_params.b_2 = self._b_2

    # ──────────────────────────────────────────────────────────────────────
    # Свойства
    # ──────────────────────────────────────────────────────────────────────

    @property
    def q(self) -> np.ndarray:
        """Вектор обобщённых координат ``[x, θ₁, θ₂]``."""
        return self._q.copy()

    @property
    def dq(self) -> np.ndarray:
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
    # Публичный API
    # ──────────────────────────────────────────────────────────────────────

    def update_physics(
        self, F_ideal: float, F_noise: NoiseForce
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
        if _co_cpp is None:
            raise RuntimeError(
                "C++ backend (co_cpp) is not available. Rebuild the pybind11 module or keep the Python fallback enabled."
            )

        # Fast path: use update_physics_cpp if available.
        if hasattr(_co_cpp, "update_physics_cpp"):
            # Pass q/dq as writable contiguous float64 buffers.
            q_arr = np.asarray(self._q, dtype=np.float64)
            dq_arr = np.asarray(self._dq, dtype=np.float64)
            if not q_arr.flags["WRITEABLE"]:
                q_arr = q_arr.copy()
            if not dq_arr.flags["WRITEABLE"]:
                dq_arr = dq_arr.copy()

            backlash_alpha = (
                float(self._backlash.alpha) if self._backslash_mode and self._backlash is not None else 0.0
            )
            backlash_m_mot = (
                float(self._backlash._m_mot) if self._backslash_mode and self._backlash is not None else 1.0
            )

            params = self._cpp_params
            _q_arr, _dq_arr, self._cpp_backlash_gap_pos = _co_cpp.update_physics_cpp(
                q_arr,
                dq_arr,
                float(F_ideal),
                float(F_noise.mean),
                float(F_noise.std),
                float(self._dt),
                params,
                bool(self._backslash_mode),
                bool(self._single_mode),
                backlash_alpha,
                backlash_m_mot,
                float(self._cpp_backlash_gap_pos),
            )

            # Keep python state as numpy float64 arrays
            self._q = np.asarray(_q_arr, dtype=np.float64)
            self._dq = np.asarray(_dq_arr, dtype=np.float64)
            return

    def reset(self) -> None:
        """Сбросить состояние модели к начальному.

        Примечание
        ----------
        Начальные значения берутся из текущих внутренних параметров
        (как было задано при конструировании).
        """
        # Используем зафиксированное начальное состояние, если оно сохранено.
        if hasattr(self, "_q_init") and hasattr(self, "_dq_init"):
            self._q = self._q_init.copy()
            self._dq = self._dq_init.copy()
            return

        # Fallback: обнуляем скорости, координаты оставляем как есть.
        self._dq = np.array([0.0, 0.0, 0.0])

    def get_clean_state(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Вернуть абсолютно чистые (неискажённые шумами датчиков)
        координаты и скорости системы.

        Returns
        -------
        tuple[State, StateDot]
            Кортеж ``(q, dq)``.
        """
        return (self._q, self._dq)
