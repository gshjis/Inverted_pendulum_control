import numpy as np
from numpy.typing import NDArray


class NoiseGenerator:
    """
    Генератор гауссовского белого шума для имитации измерительных помех.

    Parameters
    ----------
    std : NDArray[np.float64]
        Вектор среднеквадратичных отклонений (СКО) для каждого канала.
    seed : int | None
        Опциональный seed для воспроизводимости шума.
    """

    def __init__(
        self, std: NDArray[np.float64], seed: int | None = None
    ) -> None:
        self._std = np.asarray(std, dtype=np.float64)
        self._rng = np.random.default_rng(seed)

    @property
    def std(self) -> NDArray[np.float64]:
        """Вектор СКО шумовых каналов."""
        return self._std

    def generate(self) -> NDArray[np.float64]:
        """
        Сгенерировать вектор белого шума.

        Returns
        -------
        NDArray[np.float64]
            Вектор нормально распределённых случайных величин
            с нулевым мат. ожиданием и заданными СКО.
        """
        return self._rng.normal(0.0, self._std)


class SensorBlock:
    """
    Имитация измерительной подсистемы аппаратного стенда.

    Вносит в чистые координаты ОУ искажения, характерные для реальных
    физических приборов: дискретность цифровых энкодеров по уровню
    (квантование) и высокочастотные наводки в цепях (белый измерительный шум).
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

            - **encoder_resolution_1** : ``int``
                Разрядность энкодера первого звена (импульсов на оборот).
            - **encoder_resolution_2** : ``int``
                Разрядность энкодера второго звена.
            - **cart_sensor_resolution** : ``float``
                Дискретность датчика положения каретки (м).
            - **noise_std_q** : ``Array[float]``
                СКО белого шума для каналов ``[x, θ₁, θ₂]``.
            - **noise_std_dq** : ``Array[float]``
                СКО белого шума для каналов ``[ẋ, θ̇₁, θ̇₂]``.
            - **seed** : ``int``, optional
                Seed для воспроизводимости генератора шума.
        """
        # Разрешающая способность датчиков
        self._encoder_res_1: int = int(config["encoder_resolution_1"])
        self._encoder_res_2: int = int(config["encoder_resolution_2"])
        self._cart_res: float = float(config["cart_sensor_resolution"])

        # Предвычисленные шаги квантования
        self._angle_step_1: float = 2.0 * np.pi / self._encoder_res_1
        self._angle_step_2: float = 2.0 * np.pi / self._encoder_res_2
        self._cart_step: float = self._cart_res

        # Генератор измерительного шума
        noise_std_q = np.asarray(config["noise_std_q"], dtype=np.float64)   # [x, θ₁, θ₂]
        noise_std_dq = np.asarray(config["noise_std_dq"], dtype=np.float64)  # [ẋ, θ̇₁, θ̇₂]
        full_std = np.concatenate([noise_std_q, noise_std_dq])               # [x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂]

        seed: int | None = config.get("seed", None)
        self._noise_generator = NoiseGenerator(full_std, seed=seed)

    # ──────────────────────────────────────────────────────────────────────
    # Свойства
    # ──────────────────────────────────────────────────────────────────────

    @property
    def noise_generator(self) -> NoiseGenerator:
        """Внутренний генератор белого шума."""
        return self._noise_generator

    # ──────────────────────────────────────────────────────────────────────
    # Квантование
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _apply_quantization(
        value: float, resolution: int | float, sensor_type: str
    ) -> float:
        """
        Симуляция дискретности уровней датчика.

        Parameters
        ----------
        value : float
            Непрерывное физическое значение.
        resolution : int | float
            Разрядность энкодера (для угловых датчиков) или
            шаг оптической линейки (для датчика положения тележки).
        sensor_type : {"encoder", "cart"}
            Тип датчика:
            - ``"encoder"`` — угловой энкодер (пересчёт через :math:`2\\pi / \\text{resolution}`);
            - ``"cart"`` — датчик положения каретки.

        Returns
        -------
        float
            Квантованное значение в исходных единицах (рад / м).
        """
        if sensor_type == "encoder":
            step = 2.0 * np.pi / int(resolution)
            return float(np.round(value / step) * step)
        elif sensor_type == "cart":
            step = float(resolution)
            return float(np.round(value / step) * step)
        else:
            raise ValueError(
                f"Unknown sensor_type='{sensor_type}'. "
                f"Expected 'encoder' or 'cart'."
            )

    # ──────────────────────────────────────────────────────────────────────
    # Основной метод
    # ──────────────────────────────────────────────────────────────────────

    def get_telemetry(
        self, raw_q: NDArray[np.float64], raw_dq: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Преобразовать чистые координаты ОУ в зашумлённый квантованный
        вектор телеметрии.

        Алгоритм:

        1. Квантование каждой координаты из ``raw_q`` (углы — энкодер,
           положение тележки — оптическая линейка).
        2. Генерация вектора измерительного шума через ``noise_generator``.
        3. Суммирование шума с квантованными значениями.
        4. Скорости не квантуются (предполагается, что измеряются аппаратно
           тахогенераторами или аналоговыми датчиками).

        Parameters
        ----------
        raw_q : NDArray[np.float64]
            Чистый вектор координат из ОУ ``[x, θ₁, θ₂]``.
        raw_dq : NDArray[np.float64]
            Чистый вектор скоростей из ОУ ``[ẋ, θ̇₁, θ̇₂]``.

        Returns
        -------
        NDArray[np.float64]
            Вектор измеренного состояния
            ``[x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂]``.
        """
        q = np.asarray(raw_q, dtype=np.float64)
        dq = np.asarray(raw_dq, dtype=np.float64)

        # 1. Квантование координат
        x_q = self._apply_quantization(q[0], self._cart_step, "cart")
        th1_q = self._apply_quantization(q[1], self._encoder_res_1, "encoder")
        th2_q = self._apply_quantization(q[2], self._encoder_res_2, "encoder")

        # 2. Квантованный вектор + скорости (скорости не квантуются)
        quantized = np.array(
            [x_q, th1_q, th2_q, dq[0], dq[1], dq[2]],
            dtype=np.float64,
        )

        # 3. Наложение измерительного шума
        noise = self._noise_generator.generate()
        measured_state = quantized + noise

        return measured_state
