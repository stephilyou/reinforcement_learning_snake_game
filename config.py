# config.py

import torch

# Game Constants
WIDTH, HEIGHT = 400, 400  # Window size
ROWS, COLS = 20, 20        # Grid size
SQUARE_SIZE = WIDTH // COLS
FPS = 360                   # Frames per second

# Colors (RGB)
BLACK = (0, 0, 0)
GREEN = (96, 181, 89)
DARK_GREEN = (86, 163, 80)
RED = (255, 0, 0)
BLUE = (36, 86, 166)
WHITE = (255, 255, 255)

# Agent Parameters
LEARNING_RATE_DQN = 0.001
LEARNING_RATE_TABULAR = 0.01
DISCOUNT_FACTOR = 0.95
INITITAL_EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
EPISODES = 10000    # Total number of training episodes
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE = 20  # Frequency to update target network (in episodes)

# File Paths
HIGH_SCORE_FILES = {
    "random": "ra_high_score.txt",
    "tabular": "tqa_high_score.txt",
    "dqn": "dqn_high_score.txt"
}
MODEL_FILE = "dqn_model.pth"

# Action Space
ACTION_SPACE = ['STRAIGHT', 'LEFT', 'RIGHT']

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
