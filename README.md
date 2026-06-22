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

### C++ backend

Сборка C++ ядра симуляции через pybind11.

#### Linux

```bash
# 1. Подготовить build-директорию
cd packages/simulation/CO/cpp
rm -rf build
mkdir build && cd build

# 2. Запустить cmake с Python из poetry
cmake .. \
  -DPython_EXECUTABLE="$(poetry run python -c 'import sys; print(sys.executable)')" \
  -DCMAKE_BUILD_TYPE=Release

# 3. Собрать
cmake --build . -j "$(nproc)"
```

После сборки модуль появится в `packages/simulation/CO/co_cpp.so`.

**Проверка:**
```bash
poetry run python -c "
from packages.simulation.CO import co_cpp as m
print('C++ backend OK:', m)
print('Functions:', [f for f in dir(m) if not f.startswith('_')])
"
```
#### Windows
```powershell
# 1. Подготовить build-директорию
cd packages/simulation/CO/cpp
Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path build | Out-Null
cd build

# 2. Получить путь к Python из poetry
$py = & poetry run python -c "import sys; print(sys.executable)"

# 3. Запустить cmake
cmake .. `
  -DPython_EXECUTABLE="$py" `
  -DCMAKE_BUILD_TYPE=Release

# 4. Собрать
cmake --build . --config Release
```

После сборки модуль появится в `packages\simulation\CO\co_cpp.pyd`.

**Проверка:**
```powershell
poetry run python -c "from packages.simulation.CO import co_cpp as m; print('C++ backend OK:', m); print('Functions:', [f for f in dir(m) if not f.startswith('_')])"
```

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

