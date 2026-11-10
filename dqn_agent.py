"""
dqn_agent.py
Task 5: The DQN Agent.

Contains:
 - Q-network + target network
 - optimizer
 - replay buffer
 - epsilon-greedy action selection
 - Bellman-target learning step
 - periodic target network sync
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dqn_network import DQNNetwork
from replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 256,
        lr: float = 1e-4,
        gamma: float = 0.99,
        buffer_capacity: int = 100_000,
        batch_size: int = 64,
        target_update_freq: int = 1000,   # in gradient steps
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50_000,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        # Online and target networks
        self.q_network = DQNNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity, device=self.device)

        self.learn_step_counter = 0
        self.total_env_steps = 0

    # ------------------------------------------------------------------
    # Epsilon-greedy action selection
    # ------------------------------------------------------------------
    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """
        a = random action with probability epsilon
        a = argmax_a Q(s,a) with probability 1 - epsilon
        """
        if (not greedy) and (random.random() < self.epsilon):
            return random.randrange(self.action_size)

        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_network(state_t)
            action = int(torch.argmax(q_values, dim=1).item())
        return action

    def update_epsilon(self):
        """Linearly decay epsilon from epsilon_start to epsilon_end."""
        self.total_env_steps += 1
        fraction = min(1.0, self.total_env_steps / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    # ------------------------------------------------------------------
    # Storing transitions
    # ------------------------------------------------------------------
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    # ------------------------------------------------------------------
    # Learning step (Bellman update)
    # ------------------------------------------------------------------
    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return None  # not enough data yet

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Q(s, a) for the actions actually taken
        q_values = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1, keepdim=True)[0]
            # y = r + (1 - d) * gamma * max_a' Q_target(s', a')
            targets = rewards + (1.0 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    def save(self, path: str):
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "state_size": self.state_size,
            "action_size": self.action_size,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
