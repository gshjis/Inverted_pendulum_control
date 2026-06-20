# RL & Классические контроллеры для управления обратным маятником

Платформа для исследования и сравнения методов управления перевёрнутым
маятником: от классических регуляторов до алгоритмов обучения
с подкреплением (Policy Gradient, Actor-Critic).

---

## Цель проекта

Предоставить единую, воспроизводимую среду для:

- Симуляции динамики тележки с маятником (RK4, C++ backend)
- Реализации и визуализации законов управления (PID, RL, …)
- Обучения RL-агентов с логированием и чекпоинтами
- Профилирования производительности

---

## Демонстрация

<img src="output.gif" alt="Демонстрация симуляции" />

---

## Архитектура проекта

```
RL/
├── packages/
│   ├── simulation/CO/       # Ядро симуляции (физика, датчики, контроллер, такт управления)
│   ├── controllers/
│   │   ├── PID/             # ПИД-регулятор + оптимизация (Циглер-Николс, GA)
│   │   ├── REINFORCE/       # Policy Gradient на PyTorch
│   │   └── DDPG/            # Deep Deterministic Policy Gradient
│   ├── loggers/             # Визуализация (matplotlib)
│   └── GUI/                 # Pygame-интерфейс для реального времени
├── profiling/               # Скрипты профилирования (cProfile)
└── main.py                  # Точка входа (GUI + обучение)
```

### Пакеты

| Пакет | Описание |
|---|---|
| [`simulation/CO`](packages/simulation/CO) | Физическая модель (ObjectOfControl), датчики (SensorBlock), абстрактный Controller, clock_cycle. C++ backend (pybind11) |
| [`controllers/PID`](packages/controllers/PID) | ПИД-регулятор (Kp, Ki, Kd, Kx, Kdx). Оптимизация: Циглер-Николс (положение) и генетический алгоритм (угол) |
| [`controllers/REINFORCE`](packages/controllers/REINFORCE) | Policy Gradient с Normal-распределением. Gradient accumulation, baseline, clipping, чекпоинты |
| [`controllers/DDPG`](packages/controllers/DDPG) | Deep Deterministic Policy Gradient (Actor-Critic) |
| [`loggers`](packages/loggers) | Построение графиков в реальном времени (matplotlib) |
| [`GUI`](packages/simulation/GUI) | Pygame-визуализация с управлением в реальном времени |


---

## Установка

```bash
poetry install
```

### C++ backend (опционально)

```bash
cd packages/simulation/CO/cpp
mkdir build && cd build
cmake ..
make
cp co_cpp*.so ..
```

---

## Быстрый старт

### PID

```python
from packages.simulation.GUI import PendulumViewer
from packages.controllers.PID import PIDController
from packages.controllers.PID.optimizers import Genetic_PID_AngleOnly

pid = PIDController(ControllerConfig())
ga = Genetic_PID_AngleOnly()
pid.train(plant_config, sensor_config, noise, optimizer=ga, target_state=target)

window = PendulumViewer(plant, sensor_cfg, noise, controller=pid, target_state=target)
window.use()
```

