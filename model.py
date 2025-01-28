# model.py

import torch
import torch.nn as nn
import random
from collections import deque
from config import BATCH_SIZE, MEMORY_SIZE


class DQN(nn.Module):
    """Deep Q-Network with two hidden layers."""
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # First hidden layer with 128 neurons
        self.fc2 = nn.Linear(128, 128)        # Second hidden layer with 128 neurons
        self.fc3 = nn.Linear(128, output_dim) # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayMemory:
    """Experience Replay Buffer"""
    def __init__(self, capacity=MEMORY_SIZE):
        self.memory = deque(maxlen=capacity)

    def push(self, experience):
        """Save an experience to memory."""
        self.memory.append(experience)

    def sample(self, batch_size=BATCH_SIZE):
        """Sample a batch of experiences from memory."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
