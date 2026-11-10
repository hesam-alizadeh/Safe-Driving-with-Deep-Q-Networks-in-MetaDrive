"""
dqn_network.py
Task 3: The Q-network architecture.

state_size -> hidden_size -> hidden_size -> action_size
"""

import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: (batch, state_size)
        returns: (batch, action_size) -> Q(s, a0..a5)
        """
        return self.net(state)
