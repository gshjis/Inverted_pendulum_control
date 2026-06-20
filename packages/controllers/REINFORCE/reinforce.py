from __future__ import annotations

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
from pathlib import Path


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
        
        log_std = self.log_std_layer(x)
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
        
        return mu*self._max_force

    def save(self, path: str | Path) -> None:
        """Сохранить веса нейросети в файл."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), str(path))

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        config: ReinforceNetworkConfig,
        controller_config: ControllerConfig,
    ) -> Reinforce:
        """Создать агента с конфигурацией и сразу загрузить веса."""
        agent = cls(config, controller_config)
        agent.load(path)
        return agent

    def load(self, path: str | Path) -> None:
        """Загрузить веса нейросети из файла."""
        path = Path(path)
        self.net.load_state_dict(torch.load(str(path), weights_only=True))
        self.net.eval()

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
        epochs:int = 1000,
        episodes_per_epoch: int = 100,
        logger: Optional[Logger] = None,
        *,
        method_options: dict[str, Any] | None = None,
    ) -> None:
        sensor = SensorBlock(sensor_config)
        self.net.train()
        gamma = 0.99
        dt_control = self._dt
        max_steps = int(episode_max_time / dt_control)

        ckpt_dir = Path("checkpoints") / self.name.lower()
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_interval = 10

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            epoch_rewards = []
            epoch_steps = []
            epoch_loss_val = 0.0

            for episod in range(episodes_per_epoch):
                plant = ObjectOfControl(copy.deepcopy(plant_config))
                self.reset()

                rewards: list[float] = []
                F_raw = 0.0

                for step in range(max_steps):
                    J_, F_raw = clock_cycle(
                        self, plant, sensor, noise, F_raw, target_state,
                        lambda t, m: np.exp(-np.linalg.norm(t-m) * 3.0)
                    )
                    
                    rewards.append(J_)

                    if default_terminate_condition(plant):
                        rewards[-1] = -10.0
                        break

                # Discounted returns
                G = 0.0
                returns = []
                for r in reversed(rewards):
                    G = r + gamma * G
                    returns.insert(0, G)
                epoch_rewards.append(sum(rewards))
                epoch_steps.append(len(rewards))

                # Normalise
                returns_t = torch.tensor(returns, dtype=torch.float32)
                if len(returns_t) > 1:
                    returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

                # Loss за один эпизод — плоский граф через stack
                if self._log_probs:
                    log_probs_t = torch.stack(self._log_probs)
                    episode_loss = -(log_probs_t * returns_t).sum() / len(log_probs_t)
                    episode_loss.backward()          # градиенты накапливаются в .grad
                    epoch_loss_val += episode_loss.item()

            # Один шаг оптимизатора на всю эпоху
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Логирование
            total_return = np.mean(np.array(epoch_rewards))
            mean_steps = np.mean(np.array(epoch_steps))
            print(f"Episode {epoch:4d} | Return: {total_return:8.2f} Steps: {mean_steps:3.1f}")

            # Сохранение чекпоинта с метриками в имени
            if epoch % save_interval == 0 or epoch == epochs - 1:
                ckpt_name = f"epoch_{epoch:04d}_return_{total_return:+.1f}_steps_{mean_steps:.0f}.pt"
                self.save(ckpt_dir / ckpt_name)
