from dataclasses import dataclass


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
