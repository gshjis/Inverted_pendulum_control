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

def default_terminate_condition(state: ObjectOfControl) -> bool:
    """Эпизод прерывается, если маятник отклонился более чем на 40° от вертикали."""
    if abs(state.q[1] - np.pi) > np.radians(40):
        return True
    return False


class ReinforceNet(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: list[int],
        activation: str,
        output_activation: str,
    ) -> None:
        super().__init__()
        self.activation_name = activation
        self.output_activation_name = output_activation

        self.layers = nn.ModuleList()
        prev_dim = state_dim
        for hidden_dim in hidden_layers:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        # action_dim — для среднего, ещё action_dim — для log_std
        self.mu_layer = nn.Linear(prev_dim, action_dim)
        self.log_std_layer = nn.Linear(prev_dim, action_dim)

    def _get_activation(self, name: str) -> nn.Module:
        if name == "relu":
            return nn.ReLU()
        elif name == "tanh":
            return nn.Tanh()
        elif name == "sigmoid":
            return nn.Sigmoid()
        elif name == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {name}")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            x = layer(x)
            x = self._get_activation(self.activation_name)(x)
        mu = self.mu_layer(x)
        if self.output_activation_name == "tanh":
            mu = torch.tanh(mu)
        elif self.output_activation_name == "sigmoid":
            mu = torch.sigmoid(mu)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20.0, 2.0)  # для стабильности
        return mu, log_std


class Reinforce(Controller):
    def __init__(
        self, config: ReinforceNetworkConfig, controller_config: ControllerConfig
    ) -> None:
        Controller.__init__(self, controller_config)

        self.name = "REINFORCE"
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.hidden_layers = config.hidden_layers
        self.activation = config.activation
        self.learning_rate = config.learning_rate
        self.output_activation = config.output_activation

        self.net = ReinforceNet(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_layers=self.hidden_layers,
            activation=self.activation,
            output_activation=self.output_activation,
        )
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self._log_probs: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.net(x)

    def get_action(self, s_clean: np.ndarray, target_state: np.ndarray) -> float:
        x = torch.cat(
            [
                torch.from_numpy(s_clean).float(),
                torch.from_numpy(target_state).float(),
            ],
            dim=0,
        )
        mu, log_std = self.forward(x)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        self._log_probs.append(log_prob)
        return float(torch.tanh(action).item() * self._max_force)

    def reset(self) -> None:
        super().reset()
        self._log_probs: list[torch.Tensor] = []

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

        for episode in range(episodes):
            plant = ObjectOfControl(plant_config.copy())
            self.reset()

            dt_control = self._dt
            max_steps = int(episode_max_time / dt_control)

            rewards: list[float] = []

            F_raw = 0.0
            for _ in range(max_steps):
                J_, F_raw = clock_cycle(
                    self, plant, sensor, noise, F_raw, target_state,
                    lambda t, m: -0.01 * abs(target_state[1] - m[1]).sum()
                )
                rewards.append(float(J_))

                if terminate_condition is not None and terminate_condition(plant):
                    rewards[-1] += -50.0
                    break
            

            # ── REINFORCE: вычисление returns ──
            G = 0.0
            returns: list[float] = []
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)

            returns_t = torch.tensor(returns, dtype=torch.float32)
            if len(returns_t) > 1:
                returns_t = returns_t - returns_t.mean()

            # loss = -sum(log_prob * return)
            loss = torch.tensor(0.0, dtype=torch.float32)
            for log_prob, ret in zip(self._log_probs, returns_t):
                loss = loss + log_prob * ret

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            self.optimizer.step()

            print({"episode": episode, "return": sum(rewards), "loss": loss.item()})

            if episode % 100 == 0:
                print(f"Episode {episode}, Return: {sum(rewards):.2f}, Loss: {loss.item():.4f}")
