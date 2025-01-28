# agents.py

import torch
import random
import numpy as np
from model import DQN, ReplayMemory
from game import ROWS, COLS
from config import (
    INITITAL_EPSILON, EPSILON_DECAY, MIN_EPSILON,
    ACTION_SPACE, LEARNING_RATE_DQN, LEARNING_RATE_TABULAR, DISCOUNT_FACTOR,
    BATCH_SIZE, MEMORY_SIZE, device
)
import torch.nn as nn
import torch.optim as optim

class DQNAgent:
    """Agent that interacts with the game environment and learns using DQN."""
    def __init__(self):
        self.n_games = 0
        self.epsilon = INITITAL_EPSILON
        self.gamma = DISCOUNT_FACTOR
        self.memory = ReplayMemory(capacity=MEMORY_SIZE)
        self.model = DQN(input_dim=12, output_dim=len(ACTION_SPACE)).to(device)
        self.target_model = DQN(input_dim=12, output_dim=len(ACTION_SPACE)).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE_DQN)
        self.criterion = nn.SmoothL1Loss()

    def get_state(self, snake, food):
        """
        Generates a state representation for DQN.
        """
        head = snake.positions[-1]

        # Determine the direction the snake is moving
        directions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        current_dir = snake.get_direction()
        dir_idx = directions.index(current_dir)

        # Define relative directions
        left_dir = directions[(dir_idx - 1) % 4]
        right_dir = directions[(dir_idx + 1) % 4]

        # Define direction vectors
        dir_vectors = {
            'UP': (-1, 0),
            'RIGHT': (0, 1),
            'DOWN': (1, 0),
            'LEFT': (0, -1)
        }

        # Check for danger straight, left, and right
        def is_danger(direction):
            dx, dy = dir_vectors[direction]
            next_pos = (head[0] + dx, head[1] + dy)
            return int(
                next_pos[0] < 0 or next_pos[0] >= ROWS or
                next_pos[1] < 0 or next_pos[1] >= COLS or
                next_pos in snake.positions
            )

        danger_straight = is_danger(current_dir)
        danger_left = is_danger(left_dir)
        danger_right = is_danger(right_dir)

        # Food location
        food_x, food_y = food.position
        food_dir_x = food_x - head[0]
        food_dir_y = food_y - head[1]

        # Normalize distance to food
        distance = abs(food_dir_x) + abs(food_dir_y)  # Manhattan distance
        max_distance = ROWS + COLS
        normalized_distance = distance / max_distance

        # One-hot encode the current direction
        direction_one_hot = [0, 0, 0, 0]
        direction_one_hot[dir_idx] = 1

        state = [
            danger_straight,
            danger_left,
            danger_right
        ] + direction_one_hot + [
            int(food_dir_x < 0),  # Food is up
            int(food_dir_x > 0),  # Food is down
            int(food_dir_y > 0),  # Food is right
            int(food_dir_y < 0),  # Food is left
            normalized_distance
        ]

        # Debug: Ensure state vector length is 12
        assert len(state) == 12, f"Expected state vector length 12, got {len(state)}"
        return np.array(state, dtype=np.float32)

    def choose_action(self, state):
        """
        Chooses an action based on the epsilon-greedy policy.
        """
        if random.random() < self.epsilon:
            return random.randint(0, len(ACTION_SPACE) - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
                q_values = self.model(state_tensor)
                return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        """Stores the experience in replay memory."""
        self.memory.push((state, action, reward, next_state, done))

    def optimize_model(self):
        """
        Performs one step of optimization on the policy network.
        """
        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)
        batch = list(zip(*transitions))

        # Convert lists of numpy arrays to single numpy arrays
        state_batch = torch.from_numpy(np.array(batch[0])).float().to(device)  # Shape: [BATCH_SIZE, 12]
        action_batch = torch.tensor(batch[1], dtype=torch.long).unsqueeze(1).to(device)  # Shape: [BATCH_SIZE, 1]
        reward_batch = torch.tensor(batch[2], dtype=torch.float32).unsqueeze(1).to(device)  # Shape: [BATCH_SIZE, 1]
        next_state_batch = torch.from_numpy(np.array(batch[3])).float().to(device)  # Shape: [BATCH_SIZE, 12]
        done_batch = torch.tensor(batch[4], dtype=torch.float32).unsqueeze(1).to(device)  # Shape: [BATCH_SIZE, 1]

        # Compute Q(s_t, a)
        state_action_values = self.model(state_batch).gather(1, action_batch)  # Shape: [BATCH_SIZE, 1]

        # Compute V(s_{t+1}) for all next states.
        with torch.no_grad():
            next_state_values = self.target_model(next_state_batch).max(1)[0].unsqueeze(1)  # Shape: [BATCH_SIZE, 1]
            expected_state_action_values = reward_batch + (self.gamma * next_state_values * (1 - done_batch))  # Shape: [BATCH_SIZE, 1]

        # Compute Huber loss
        loss = self.criterion(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        """Updates the target network with the policy network's weights."""
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        """Decays the exploration rate epsilon."""
        if self.epsilon > MIN_EPSILON:
            self.epsilon *= EPSILON_DECAY
            self.epsilon = max(MIN_EPSILON, self.epsilon)

import random
from config import ACTION_SPACE, INITITAL_EPSILON, EPSILON_DECAY, MIN_EPSILON

class RandomAgent:
    """Agent that selects actions randomly without learning."""
    def __init__(self):
        random.seed(42)
        self.actions = ACTION_SPACE

    def choose_action(self, state):
        return random.randint(0, len(self.actions) - 1)

class TabularQAgent:
    """Agent that interacts with the game environment and learns using Tabular Q-learning."""
    def __init__(self):
        self.epsilon = INITITAL_EPSILON
        self.discountFactor = DISCOUNT_FACTOR
        self.learningRate = LEARNING_RATE_TABULAR 
        self.q_table = {} 
        self.actions = ACTION_SPACE
        self.epsilon_decay = EPSILON_DECAY
        self.min_epsilon = MIN_EPSILON

    def get_state(self, snake, food):
        head = snake.positions[-1]

        # Determine the direction the snake is moving
        directions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        current_dir = snake.get_direction()
        dir_idx = directions.index(current_dir)

        # Define relative directions
        left_dir = directions[(dir_idx - 1) % 4]
        right_dir = directions[(dir_idx + 1) % 4]

        # Define direction vectors
        dir_vectors = {
            'UP': (-1, 0),
            'RIGHT': (0, 1),
            'DOWN': (1, 0),
            'LEFT': (0, -1)
        }

        # Check for danger straight, left, and right
        def is_danger(direction):
            dx, dy = dir_vectors[direction]
            next_pos = (head[0] + dx, head[1] + dy)
            return int(
                next_pos[0] < 0 or next_pos[0] >= ROWS or
                next_pos[1] < 0 or next_pos[1] >= COLS or
                next_pos in snake.positions
            )

        danger_straight = is_danger(current_dir)
        danger_left = is_danger(left_dir)
        danger_right = is_danger(right_dir)

        # Danger code: combine danger flags into a single integer (0-7)
        danger_code = danger_straight * 4 + danger_left * 2 + danger_right * 1

        # Food location
        food_x, food_y = food.position
        food_dir_x = food_x - head[0]
        food_dir_y = food_y - head[1]

        # Food direction: encode into 8 possible directions
        if food_dir_x < 0 and food_dir_y == 0:
            food_direction = 0  # Up
        elif food_dir_x < 0 and food_dir_y > 0:
            food_direction = 1  # Up-Right
        elif food_dir_x == 0 and food_dir_y > 0:
            food_direction = 2  # Right
        elif food_dir_x > 0 and food_dir_y > 0:
            food_direction = 3  # Down-Right
        elif food_dir_x > 0 and food_dir_y == 0:
            food_direction = 4  # Down
        elif food_dir_x > 0 and food_dir_y < 0:
            food_direction = 5  # Down-Left
        elif food_dir_x == 0 and food_dir_y < 0:
            food_direction = 6  # Left
        elif food_dir_x < 0 and food_dir_y < 0:
            food_direction = 7  # Up-Left
        else:
            food_direction = 0  # If the food is on the head, treat as Up

        # Construct the state tuple
        state = (danger_code, dir_idx, food_direction)
        return state

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, len(self.actions) - 1)
        else:
            # Choose action with highest Q-value
            q_values = [self.q_table.get((state, action), 0) for action in range(len(self.actions))]
            max_q = max(q_values)
            max_actions = [i for i, q in enumerate(q_values) if q == max_q]
            return random.choice(max_actions)

    def learn(self, state, action, reward, next_state, done):
        current_q = self.q_table.get((state, action), 0)
        if done or next_state is None:
            target_q = reward
        else:
            # Estimate the optimal future value
            next_q_values = [self.q_table.get((next_state, a), 0) for a in range(len(self.actions))]
            max_next_q = max(next_q_values)
            target_q = reward + self.discountFactor * max_next_q

        # Update Q-value
        self.q_table[(state, action)] = current_q + self.learningRate * (target_q - current_q)

    def decay_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.min_epsilon, self.epsilon)
