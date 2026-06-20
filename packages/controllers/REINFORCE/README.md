# REINFORCE — Policy Gradient (RL)

Агент алгоритма REINFORCE (Monte-Carlo Policy Gradient) для управления
перевёрнутым маятником.

---

## Обзор

Реализует стохастическую политику (Normal-распределение) с нейросетью
на PyTorch:

$$\pi_\theta(a|s) = \mathcal{N}(\mu_\theta(s), \sigma_\theta(s))$$

- **Архитектура сети**: линейные слои + activation → `mu` и `log_std`
- **Обучение**: метод REINFORCE с gradient accumulation, baseline
  и gradient clipping
- **Чекпоинты**: автоматическое сохранение каждые N эпох

---

## Зависимости

- `packages/simulation/CO` — контроллер, физика, датчики, такт управления
- `torch ≥ 2.4` — нейросеть и оптимизатор

---

## Быстрый старт

```python
import numpy as np
from packages.simulation.CO.datatypes import *
from packages.controllers.REINFORCE.reinforce import Reinforce
from packages.controllers.REINFORCE.mode_config import ReinforceNetworkConfig

net_cfg = ReinforceNetworkConfig(
    state_dim=12,          # s_clean (6) + target_state (6)
    action_dim=1,          # сила F
    hidden_layers=[64, 64],
    learning_rate=1e-4,
)
ctrl_cfg = ControllerConfig(dt=0.005, max_force=30.0)

agent = Reinforce(net_cfg, ctrl_cfg)
agent.train(
    plant_config=PlantConfig(),
    sensor_config=SensorConfig(),
    noise=NoiseForce(mean=0.0, std=0.0),
    target_state=np.array([0.0, np.pi, 0.0, 0.0, 0.0, 0.0]),
    episode_max_time=20.0,
    epochs=500,
    episodes_per_epoch=100,
)

# Сохранение
agent.save("checkpoints/reinforce/final.pt")

# Загрузка
agent = Reinforce.from_pretrained("checkpoints/reinforce/final.pt", net_cfg, ctrl_cfg)
```

---

## Компоненты

| Класс | Файл | Описание |
|---|---|---|
| `ReinforceNetworkConfig` | `mode_config.py` | Конфигурация сети (state_dim, hidden, lr, активация) |
| `ReinforceNet` | `reinforce.py` | Нейросеть: линейные слои → mu / log_std |
| `Reinforce` | `reinforce.py` | Агент: Controller + обучение + save/load |

**Основные методы `Reinforce`:**

| Метод | Описание |
|---|---|
| `get_action(s_clean, target)` | Сэмплировать действие из Normal(mu, std) |
| `train(plant_config, sensor_config, ...)` | Многоэпизодное обучение с чекпоинтами |
| `save(path)` | Сохранить state_dict сети |
| `load(path)` | Загрузить веса |
| `from_pretrained(path, config, ...)` | Создать агента и загрузить веса |