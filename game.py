# game.py

import pygame as pg
import random
from collections import deque
from config import WIDTH, HEIGHT, ROWS, COLS, SQUARE_SIZE, GREEN, DARK_GREEN, RED, BLUE, WHITE, BLACK

pg.init()
pg.font.init()
BOARD_Y_OFFSET = 50


class Board:
    """Handles the game board and rendering."""
    AGENT_FULL_NAMES = {
        'dqn': 'Deep Q-Network',
        'random': 'Random Agent',
        'tabular': 'Tabular Q-Learning'
    }
    TEXT_AREA_Y = 10
    AGNET_NAME_AREA_X = 610
    SCORE_AREA_X = 10
    
    # Draws a checkered grid on the game window.
    @staticmethod
    def draw_checkered_grid(screen):
        for row in range(ROWS):
            for col in range(COLS):
                color = GREEN if (row + col) % 2 == 0 else DARK_GREEN
                pg.draw.rect(
                    screen,
                    color,
                    (
                        col * SQUARE_SIZE,
                        BOARD_Y_OFFSET + row * SQUARE_SIZE,  # Apply vertical offset
                        SQUARE_SIZE,
                        SQUARE_SIZE
                    )
                )

    # Displays the current score and high score on the game window.
    @staticmethod
    def draw_score(screen, points, high_score, episode, agent_type):
        font = pg.font.Font(None, 36)
        text_y = Board.TEXT_AREA_Y
        agent_name_text_x = Board.AGNET_NAME_AREA_X
        score_text_x = Board.SCORE_AREA_X
        score_text = font.render(f"Score: {points}", True, WHITE)
        high_score_text = font.render(f"High Score: {high_score}", True, WHITE)
        episode_text = font.render(f"Iteration: {episode}", True, WHITE)
        agent_full_name = Board.AGENT_FULL_NAMES.get(agent_type, 'Unknown Agent')
        agent_label = font.render(f"{agent_full_name}", True, WHITE)
        screen.blit(agent_label, (agent_name_text_x, text_y))
        screen.blit(score_text, (score_text_x, text_y))
        screen.blit(high_score_text, (WIDTH - high_score_text.get_width() - 10, text_y))
        screen.blit(episode_text, (WIDTH+850 - episode_text.get_width() - 10, text_y))


class Food:
    """Handles food placement and rendering."""
    def __init__(self):
        self.color = RED
        self.position = None
        self.relocate([])

    # Draws the food on the game window.
    def draw(self, screen):
        if self.position:
            pg.draw.rect(
                screen,
                self.color,
                (
                    self.position[0] * SQUARE_SIZE,
                    BOARD_Y_OFFSET + self.position[1] * SQUARE_SIZE,  # Apply vertical offset
                    SQUARE_SIZE,
                    SQUARE_SIZE
                )
            )

    # Relocates the food to a random position not occupied by the snake.
    def relocate(self, positions):
        available_positions = [(x, y) for x in range(COLS) for y in range(ROWS) if (x, y) not in positions]
        if available_positions:
            self.position = random.choice(available_positions)
        else:
            # If no available positions, the snake has filled the board; win condition
            self.position = None


class Snake:
    """Handles the snake's behavior and rendering."""
    def __init__(self):
        self.color = BLUE
        self.length = 3
        self.positions = deque([(ROWS//2, COLS//2 + i) for i in range(3)])  # Initial body positions
        self.direction = (0, -1)  # Initial direction (moving left)

    # Draws the snake on the game window.
    def draw(self, screen):
        for position in self.positions:
            pg.draw.rect(
                screen,
                self.color,
                (
                    position[0] * SQUARE_SIZE,
                    BOARD_Y_OFFSET + position[1] * SQUARE_SIZE,  # Apply vertical offset
                    SQUARE_SIZE,
                    SQUARE_SIZE
                )
            )

    # Moves the snake in the given direction based on the action.
    def move(self, action):
        # Define direction mappings
        directions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        idx = directions.index(self.get_direction())

        if action == 'LEFT':
            idx = (idx - 1) % 4
        elif action == 'RIGHT':
            idx = (idx + 1) % 4
        # 'STRAIGHT' keeps the current direction

        new_direction = directions[idx]

        # Prevent the snake from reversing direction
        opposite_directions = {
            'UP': 'DOWN',
            'DOWN': 'UP',
            'LEFT': 'RIGHT',
            'RIGHT': 'LEFT'
        }
        if new_direction == opposite_directions[self.get_direction()]:
            new_direction = self.get_direction()  # Ignore the action to prevent reversal

        self.set_direction(new_direction)

        dx, dy = self.direction
        current_head = self.positions[-1]
        new_head = (current_head[0] + dx, current_head[1] + dy)
        self.positions.append(new_head)
        if len(self.positions) > self.length:
            self.positions.popleft()

    # Returns the current direction as a string.
    def get_direction(self):
        if self.direction == (0, -1):
            return 'LEFT'
        elif self.direction == (0, 1):
            return 'RIGHT'
        elif self.direction == (-1, 0):
            return 'UP'
        elif self.direction == (1, 0):
            return 'DOWN'

    # Sets the snake's direction based on a string.
    def set_direction(self, direction):
        if direction == 'LEFT':
            self.direction = (0, -1)
        elif direction == 'RIGHT':
            self.direction = (0, 1)
        elif direction == 'UP':
            self.direction = (-1, 0)
        elif direction == 'DOWN':
            self.direction = (1, 0)

    # Checks if the snake has collided with the walls.
    def check_wall_collisions(self):
        current_head = self.positions[-1]
        return (current_head[0] < 0 or current_head[0] >= ROWS or
                current_head[1] < 0 or current_head[1] >= COLS)

    # Checks if the snake has collided with the food.
    def check_food_collisions(self, food_position):
        return self.positions[-1] == food_position

    # Checks if the snake has collided with itself.
    def check_self_collisions(self):
        return self.positions[-1] in list(self.positions)[:-1]
