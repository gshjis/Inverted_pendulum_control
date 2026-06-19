from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ReinforceNetworkConfig:
    """Конфигурация архитектуры нейронной сети для алгоритма REINFORCE."""

    state_dim: int
    action_dim: int
    hidden_layers: List[int] = field(default_factory=lambda: [64, 64])
    activation: str = "relu"
    learning_rate: float = 1e-2
    output_activation: str = "tanh"
