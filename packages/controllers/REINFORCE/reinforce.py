from controller import Controller
from datatypes import ControllerConfig
from packages.controllers.REINFORCE.mode_config import ReinforceNetworkConfig
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from packages.simulation.CO import (
    ObjectOfControl,
    PlantConfig,
    SensorBlock,
    SensorConfig,
    NoiseForce,
    clock_cycle,
)
from typing import Callable, Optional, Any
from loggers import Logger
import copy


def default_terminate_condition(state: ObjectOfControl) -> bool:
    """Падение если маятник отклонился от π более чем на 40°."""
    angle = state.q[1]
    # Берем угол по модулю 2π
    angle = angle % (2 * np.pi)
    # Проверяем отклонение от π
    deviation = abs(angle - np.pi)
    # Учитываем, что отклонение может быть через 0
    deviation = min(deviation, 2*np.pi - deviation)
    return deviation > np.radians(40) or abs(state.q[0]) > 3


class ReinforceNet(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: list[int],
        activation: str = "tanh",
        output_activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.activation_name = activation
        self.output_activation_name = output_activation

        self.layers = nn.ModuleList()
        prev_dim = state_dim
        for hidden_dim in hidden_layers:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        self.mu_layer = nn.Linear(prev_dim, action_dim)
        self.log_std_layer = nn.Linear(prev_dim, action_dim)

    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "gelu": nn.GELU(),
        }
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}")
        return activations[name]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            x = self._get_activation(self.activation_name)(layer(x))
        
        mu = self.mu_layer(x)
        if self.output_activation_name == "tanh":
            mu = torch.tanh(mu)
        elif self.output_activation_name == "sigmoid":
            mu = torch.sigmoid(mu)
        
        log_std = torch.clamp(self.log_std_layer(x), -5.0, 2.0)
        return mu, log_std


class Reinforce(Controller):
    def __init__(
        self, config: ReinforceNetworkConfig, controller_config: ControllerConfig
    ) -> None:
        Controller.__init__(self, controller_config)

        self.name = "REINFORCE"
        self.net = ReinforceNet(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_layers=config.hidden_layers,
            activation=config.activation,
            output_activation=config.output_activation,
        )
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config.learning_rate)
        self._log_probs: list[torch.Tensor] = []
        self._max_force = controller_config.max_force

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.net(x)

    def get_action(self, s_clean: np.ndarray, target_state: np.ndarray) -> float:
        x = torch.cat([
            torch.from_numpy(s_clean).float(),
            torch.from_numpy(target_state).float(),
        ], dim=0)
        
        mu, log_std = self.forward(x)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        self._log_probs.append(log_prob)
        
        return float(action.item() * self._max_force)

    def reset(self) -> None:
        super().reset()
        self._log_probs = []

    def train(
        self,
        plant_config: PlantConfig,
        sensor_config: SensorConfig,
        noise: NoiseForce,
        target_state: np.ndarray,
        terminate_condition: Callable[[ObjectOfControl], bool] | None = None,
        episode_max_time: float = 150.0,
        episodes: int = 1000,
        logger: Optional[Logger] = None,
        *,
        method_options: dict[str, Any] | None = None,
    ) -> None:
        sensor = SensorBlock(sensor_config)
        self.net.train()
        gamma = 0.99
        dt_control = self._dt
        max_steps = int(episode_max_time / dt_control)

        for episode in range(episodes):
            # Создаём свежий экземпляр маятника
            plant = ObjectOfControl(copy.deepcopy(plant_config))
            self.reset()

            rewards: list[float] = []
            F_raw = 0.0

            for _ in range(max_steps):
                J_, F_raw = clock_cycle(
                    self, plant, sensor, noise, F_raw, target_state,
                    lambda t, m: 0.0  # отключаем старую награду
                )
                
                # Новая награда: +1 за выживание
                rewards.append(1.0)

                # Если маятник упал — завершаем эпизод
                if default_terminate_condition(plant):
                    rewards[-1] = -100.0  # Штраф за падение
                    break

            # Вычисляем discounted returns
            G = 0.0
            returns = []
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)

            # Нормализация returns
            returns_t = torch.tensor(returns, dtype=torch.float32)
            if len(returns_t) > 1:
                returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

            # Вычисляем loss (максимизируем log_prob * ret)
            loss = torch.tensor(0.0, dtype=torch.float32)
            for log_prob, ret in zip(self._log_probs, returns_t):
                loss = loss - log_prob * ret  # стандартный REINFORCE
            loss = loss / max(1, len(self._log_probs))

            # Оптимизация
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Логирование
            total_return = sum(rewards)
            if episode % 10 == 0:
                print(f"Episode {episode:4d} | Return: {total_return:8.2f} | Loss: {loss.item():8.4f} | Steps: {len(rewards):3d}")

            if logger is not None and episode % 100 == 0:
                logger.log_metrics({
                    "episode": episode,
                    "return": total_return,
                    "loss": loss.item(),
                    "steps": len(rewards)
                })