from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim.adam import Adam
import numpy as np
from numpy.typing import NDArray

from packages.simulation.CO import (
    Controller,
    ControllerConfig,
    MeasuredState,
    NoiseForce,
    ObjectOfControl,
    PlantConfig,
    SensorBlock,
    SensorConfig,
    State,
)

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, a):
        return self.net(torch.cat([x, a], dim=1))

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = []
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)).unsqueeze(1),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1)
        )

class DDPGController(Controller):
    def __init__(self, config: ControllerConfig) -> None:
        super().__init__(config)
        self.state_dim = 6
        self.action_dim = 1
        self.actor = Actor(self.state_dim, self.action_dim)
        self.critic = Critic(self.state_dim, self.action_dim)
        self.target_actor = Actor(self.state_dim, self.action_dim)
        self.target_critic = Critic(self.state_dim, self.action_dim)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=1e-3)
        self.buffer = ReplayBuffer(10000)

    def get_action(self, s_clean: MeasuredState, target_state: State) -> float:
        state_vec = np.array([
            s_clean.x - target_state.x,
            s_clean.theta1 - target_state.theta1,
            s_clean.theta2 - target_state.theta2,
            s_clean.x_dot,
            s_clean.theta1_dot,
            s_clean.theta2_dot
        ], dtype=np.float32)
        
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0)
        action = self.actor(state_tensor).item()
        return action * self._max_force

    def train(self, plant_config: PlantConfig, sensor_config: SensorConfig, target_state: State) -> None:
        plant = ObjectOfControl(plant_config)
        sensor = SensorBlock(sensor_config)
        noise = NoiseForce(0.0)
        
        batch_size = 64
        gamma = 0.99
        tau = 0.005

        for episode in range(100):
            plant = ObjectOfControl(plant_config)
            self.reset()
            total_reward = 0
            
            for step in range(200):
                measured = sensor.get_telemetry(plant.q, plant.dq)
                ms = MeasuredState(
                    x=measured[0], theta1=measured[1], theta2=measured[2],
                    x_dot=measured[3], theta1_dot=measured[4], theta2_dot=measured[5]
                )
                
                state_vec = np.array([
                    ms.x - target_state.x,
                    ms.theta1 - target_state.theta1,
                    ms.theta2 - target_state.theta2,
                    ms.x_dot,
                    ms.theta1_dot,
                    ms.theta2_dot
                ], dtype=np.float32)
                
                action = self.get_action(ms, target_state) / self._max_force
                action += np.random.normal(0, 0.1)
                
                plant.update_physics(action * self._max_force, noise, 0.005)
                
                next_measured = sensor.get_telemetry(plant.q, plant.dq)
                next_ms = MeasuredState(
                    x=next_measured[0], theta1=next_measured[1], theta2=next_measured[2],
                    x_dot=next_measured[3], theta1_dot=next_measured[4], theta2_dot=next_measured[5]
                )
                
                next_state_vec = np.array([
                    next_ms.x - target_state.x,
                    next_ms.theta1 - target_state.theta1,
                    next_ms.theta2 - target_state.theta2,
                    next_ms.x_dot,
                    next_ms.theta1_dot,
                    next_ms.theta2_dot
                ], dtype=np.float32)
                
                reward = - (next_ms.theta1 - target_state.theta1)**2
                done = abs(next_ms.theta1 - target_state.theta1) > np.radians(15)
                
                self.buffer.push(state_vec, action, reward, next_state_vec, done)
                total_reward += reward
                
                if len(self.buffer.buffer) > batch_size:
                    states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
                    
                    # Critic update
                    target_actions = self.target_actor(next_states)
                    target_q = self.target_critic(next_states, target_actions)
                    y = rewards + gamma * target_q * (1 - dones)
                    q = self.critic(states, actions)
                    critic_loss = nn.MSELoss()(q, y.detach())
                    
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    self.critic_optimizer.step()
                    
                    # Actor update
                    actor_loss = -self.critic(states, self.actor(states)).mean()
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()
                    
                    # Soft update
                    for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                    for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
            print(f"[Episode {episode:>3d}]  Total Reward: {total_reward:>10.4f}")
