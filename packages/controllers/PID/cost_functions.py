import numpy as np


def J(targ: np.ndarray, real: np.ndarray) -> float:
    """
    Квадратичная целевая функция (стоимость).

    Вычисляет сумму квадратов отклонений между целевым и реальным
    состоянием: :math:`J = \\|\\text{targ} - \\text{real}\\|^2`.

    Parameters
    ----------
    targ : np.ndarray
        Целевой вектор состояния (6,).
    real : np.ndarray
        Реальный (измеренный) вектор состояния (6,).

    Returns
    -------
    float
        Сумма квадратов отклонений.

    Notes
    -----
    Optimization potential:
        - Для высокочастотного вызова можно заменить ``np.dot``
          на ``np.sum(diff ** 2)`` — разница незначительна.
        - Если размерность фиксирована (6), можно развернуть цикл
          вручную для микро-оптимизации.
    """
    diff = targ - real
    return np.dot(diff, diff)