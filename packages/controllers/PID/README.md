# PID — ПИД-регулятор (baseline)

Пакет с реализацией ПИД-регулятора для управления перевёрнутым маятником
и оптимизаторами его коэффициентов.

---

## 1. Обзор

Пакет предоставляет:

- **`PIDController`** — ПИД-регулятор с демпфированием по положению
  и скорости тележки (5 коэффициентов: Kp, Ki, Kd, Kx, Kdx)
- **`Zigler_Nikols`** — метод Циглера-Николса для подбора Kx/Kdx
- **`Genetic_PID_AngleOnly`** — генетический алгоритм для подбора Kp/Ki/Kd
- **`J()`** — квадратичная целевая функция (сумма квадратов ошибок)

Пакет зависит от `packages/simulation/CO` (Controller, clock_cycle, ...).

---

## 2. Установка

```bash
poetry install
```

---

## 3. Компоненты

### 3.1. [`PIDController`](pid.py)

Закон управления:

$$u = K_p e_\theta + K_i \int e_\theta dt + K_d \dot{e}_\theta + K_x e_x + K_{dx} \dot{e}_x$$

где $e = target\_state - s\_clean$.

**Конструктор:**
```python
def __init__(self, config: ControllerConfig, gains: np.ndarray | None = None) -> None
```

| Параметр | Тип | Описание |
|---|---|---|
| `config` | `ControllerConfig` | Конфигурация регулятора (dt, max_force, фильтры) |
| `gains` | `np.ndarray \| None` | `[Kp, Ki, Kd, Kx, Kdx]`; `None` → `[10, 1, 2, 1, 2]` |

**Методы:**

| Метод | Возврат | Описание |
|---|---|---|
| `get_action(s_clean, target)` | `float` | ПИД-закон управления |
| `gains` (property) | `NDArray[5]` | Текущие коэффициенты |
| `gains` (setter) | — | Установка коэффициентов |
| `reset_angel_integral()` | — | Сброс интеграла |
| `reset()` | — | Сброс + интеграл |
| `_run_episode(plant, sensor, ...)` | `(J, time)` | Прогон одного эпизода |
| `train(plant_config, sensor_config, ...)` | — | Запуск оптимизации |

### 3.2. [`J()`](cost_functions.py) — целевая функция

$$J(target, measured) = \|target - measured\|^2$$

Импортируется как `cost_functions.J`.

---

## 4. Оптимизаторы

### 4.1. Zigler_Nikols

*Описание будет добавлено.*

### 4.2. Genetic_PID_AngleOnly

*Описание будет добавлено.*

---

## 5. Пример использования

```python
import numpy as np
from packages.simulation.CO import (
    PlantConfig, SensorConfig, ControllerConfig, NoiseForce,
    ObjectOfControl, SensorBlock,
)
from packages.controllers.PID import PIDController, terminate_condition
from packages.controllers.PID.optimizers import Zigler_Nikols, Genetic_PID_AngleOnly

# Конфигурации
plant_cfg = PlantConfig()
sensor_cfg = SensorConfig()
ctrl_cfg = ControllerConfig(dt=0.005, max_force=30.0)
noise = NoiseForce(mean=0.0, std=0.0)
target = np.array([0.0, np.pi, 0.0, 0.0, 0.0, 0.0])

# Создаём PID
pid = PIDController(ctrl_cfg, gains=np.array([10.0, 0.0, 2.0, 0.0, 0.0]))

# Оптимизация положения (Циглер-Николс)
zn = Zigler_Nikols()
result_kx = pid.train(
    plant_config=plant_cfg,
    sensor_config=sensor_cfg,
    noise=noise,
    optimizer=zn,
    target_state=target,
    terminate_condition=terminate_condition,
    episode_max_time=30.0,
)
print(result_kx)

# Оптимизация угла (GA)
ga = Genetic_PID_AngleOnly()
result_kp = pid.train(
    plant_config=plant_cfg,
    sensor_config=sensor_cfg,
    noise=noise,
    optimizer=ga,
    target_state=target,
    terminate_condition=terminate_condition,
    episode_max_time=30.0,
)
print(result_kp)
```

---

## 6. API Reference

| Модуль | Класс / Функция | Описание |
|---|---|---|
| `pid` | `PIDController` | ПИД-регулятор (5 коэф.) |
| `pid` | `terminate_condition` | Условие завершения (всегда False) |
| `cost_functions` | `J` | Квадратичная стоимость |
| `optimizers` | `Zigler_Nikols` | Оптимизация Kx/Kdx (Циглер-Николс) |
| `optimizers` | `Genetic_PID_AngleOnly` | Оптимизация Kp/Ki/Kd (GA) |