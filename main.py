# main.py

import sys
import argparse
import pygame as pg
import numpy as np
import torch

from game import ROWS, COLS, Snake, Food, Board
from config import (
    ACTION_SPACE, EPISODES, TARGET_UPDATE, 
    HIGH_SCORE_FILES, MODEL_FILE, FPS
)
from agents import DQNAgent, RandomAgent, TabularQAgent
from helper import save_high_score, reset_high_score

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (196, 202, 255)
ORANGE = (255, 165, 0)
GREEN = (190, 247, 195)
RED = (255, 0, 0)

def initialize_pygame_window():
    """Initializes the Pygame window with a fixed size."""
    pg.display.set_caption("Snake Game with RL Agents")
    window_width, window_height = 1250, 500
    screen = pg.display.set_mode((window_width, window_height))
    return screen

def define_areas():
    """Defines the game board, text, and plot areas."""
    game_board_rect = pg.Rect(0, 0, 450, 500)     # Game board area
    text_area_rect = pg.Rect(0, 0, 1250, 50)      # Text area at the top
    plot_area1 = pg.Rect(450, 50, 400, 400)       # Left plot
    plot_area2 = pg.Rect(850, 50, 400, 400)       # Right plot
    return game_board_rect, text_area_rect, plot_area1, plot_area2

def draw_plot(screen, plot_area, data, metrics, point_metrics=None):
    """Draws specified metrics on a single plot within Pygame."""
    if point_metrics is None:
        point_metrics = []

    # Clear plot area
    pg.draw.rect(screen, WHITE, plot_area)

    # Define margins and plot dimensions
    margin = 60
    plot_width = plot_area.width - 2 * margin
    plot_height = plot_area.height - 2 * margin

    # Draw axes
    pg.draw.line(screen, BLACK, 
                 (plot_area.x + margin, plot_area.y + margin), 
                 (plot_area.x + margin, plot_area.y + plot_height + margin), 2)  # Y-axis
    pg.draw.line(screen, BLACK, 
                 (plot_area.x + margin, plot_area.y + plot_height + margin), 
                 (plot_area.x + plot_width + margin, plot_area.y + plot_height + margin), 2)  # X-axis

    # Return if insufficient data
    if len(data['episodes']) < 2:
        return

    # Determine y-axis range
    min_y = min(min(data[metric], default=0) for metric in metrics)
    max_y = max(max(data[metric], default=1) for metric in metrics)
    min_y = min_y * 1.1 if min_y < 0 else 0
    scale_y = plot_height / (max_y - min_y) if max_y != min_y else 1
    scale_x = plot_width / max(data['episodes'])

    # Mapping functions
    def map_x(episode):
        return plot_area.x + margin + episode * scale_x

    def map_y(value):
        return plot_area.y + plot_height + margin - (value - min_y) * scale_y

    # Draw y-axis ticks and labels
    font = pg.font.SysFont(None, 20)
    num_y_ticks = 5
    y_tick_interval = (max_y - min_y) / num_y_ticks
    for i in range(num_y_ticks + 1):
        y_val = min_y + i * y_tick_interval
        y_pos = map_y(y_val)
        pg.draw.line(screen, (200, 200, 200), 
                     (plot_area.x + margin, y_pos), 
                     (plot_area.x + plot_width + margin, y_pos), 1)  # Grid lines
        pg.draw.line(screen, BLACK, 
                     (plot_area.x + margin - 5, y_pos), 
                     (plot_area.x + margin + 5, y_pos), 2)  # Tick marks
        label = font.render(f"{y_val:.1f}", True, BLACK)
        screen.blit(label, (plot_area.x + margin - 50, y_pos - 10))

    # Draw x-axis ticks and labels
    num_x_ticks = 5
    x_tick_interval = max(data['episodes']) / num_x_ticks
    for i in range(num_x_ticks + 1):
        x_val = i * x_tick_interval
        x_pos = map_x(x_val)
        pg.draw.line(screen, (200, 200, 200), 
                     (x_pos, plot_area.y + margin), 
                     (x_pos, plot_area.y + plot_height + margin), 1)  # Grid lines
        pg.draw.line(screen, BLACK, 
                     (x_pos, plot_area.y + plot_height + margin - 5), 
                     (x_pos, plot_area.y + plot_height + margin + 5), 2)  # Tick marks
        label = font.render(f"{int(x_val)}", True, BLACK)
        screen.blit(label, (x_pos - 10, plot_area.y + plot_height + margin + 10))

    # Define colors for metrics
    metric_colors = {
        'scores': BLUE,
        'high_scores': ORANGE,
        'rewards': GREEN,
        'avg_rewards': RED
    }

    # Draw metrics
    for metric in metrics:
        color = metric_colors.get(metric, BLACK)
        if metric in point_metrics:
            for i, episode in enumerate(data['episodes']):
                x = map_x(episode)
                y = map_y(data[metric][i])
                pg.draw.circle(screen, color, (int(x), int(y)), 3)
        else:
            points = list(zip(data['episodes'], data[metric]))
            for (x1, y1), (x2, y2) in zip(points[:-1], points[1:]):
                pg.draw.line(screen, color, (map_x(x1), map_y(y1)), (map_x(x2), map_y(y2)), 2)

    # Draw metric labels
    y_offset = 10
    for metric in metrics:
        label = font.render(metric.replace('_', ' ').title(), True, metric_colors.get(metric, BLACK))
        screen.blit(label, (plot_area.x + margin + 10, plot_area.y + y_offset))
        y_offset += 20

def handle_events():
    """Handles Pygame events to keep the window responsive."""
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            sys.exit()

def update_game_screen(screen, game_board_rect, text_area_rect, snake, food, points, high_score, episode, agent_type):
    """Renders the game state on the screen."""
    screen.fill(BLACK, game_board_rect)
    screen.fill(BLACK, text_area_rect)
    Board.draw_checkered_grid(screen)
    food.draw(screen)
    snake.draw(screen)
    Board.draw_score(screen, points, high_score, episode, agent_type)
    pg.display.update()

def save_model_periodically(agent, episode):
    """Saves the agent's model at specified intervals."""
    if episode % 100 == 0:
        torch.save(agent.model.state_dict(), MODEL_FILE)

def print_progress(agent_type, episode, points, high_score, avg_reward, agent=None):
    """Prints training progress to the console."""
    if agent_type in ['dqn', 'tabular']:
        epsilon_info = f", Epsilon: {agent.epsilon:.4f}" if agent else ""
        print(f"Episode: {episode}, Points: {points}, High Score: {high_score}{epsilon_info}, "
              f"Average Reward (last 100): {avg_reward:.2f}")
    else:
        print(f"Episode: {episode}, Points: {points}, High Score: {high_score}, "
              f"Average Reward (last 100): {avg_reward:.2f}")

def generic_training_loop(agent, high_score_file, screen, game_board_rect, text_area_rect, plot_area1, plot_area2, agent_type):
    """Generic training loop for different agent types."""
    clock = pg.time.Clock()
    high_score = reset_high_score(high_score_file)
    rewards_history = []
    data = {
        'episodes': [],
        'scores': [],
        'high_scores': [],
        'rewards': [],
        'avg_rewards': []
    }

    for episode in range(1, EPISODES + 1):
        snake = Snake()
        food = Food()
        points = 0
        state = agent.get_state(snake, food) if agent_type != 'random' else None
        done = False

        episode_reward = 0
        steps = 0

        while not done and steps < 1000:
            handle_events()

            # Action selection
            if agent_type == 'random':
                action_idx = agent.choose_action(state)
            else:
                action_idx = agent.choose_action(state)
            action = ACTION_SPACE[action_idx]

            # Move the snake
            snake.move(action)

            # Collision and reward handling
            reward = 0
            if snake.check_food_collisions(food.position):
                snake.length += 1
                points += 1
                reward = 10
                food.relocate(snake.positions)
            else:
                reward = -0.1

            if food.position:
                current_distance = abs(food.position[0] - snake.positions[-1][0]) + \
                                   abs(food.position[1] - snake.positions[-1][1])
                reward += -0.1 * (current_distance / (ROWS + COLS))
            else:
                reward = -10
                done = True

            if snake.check_wall_collisions() or snake.check_self_collisions():
                reward = -10
                done = True
                if points > high_score:
                    high_score = points
                    save_high_score(high_score, high_score_file)

            # Next state determination
            if not done:
                next_state = agent.get_state(snake, food) if agent_type != 'random' else None
            else:
                next_state = np.zeros(12, dtype=np.float32) if agent_type == 'dqn' else None

            # Agent-specific learning
            if agent_type == 'dqn':
                agent.remember(state, action_idx, reward, next_state, done)
                agent.optimize_model()
            elif agent_type == 'tabular':
                agent.learn(state, action_idx, reward, next_state, done)

            # Move to next state
            state = next_state

            # Accumulate rewards and steps
            episode_reward += reward
            steps += 1

            # Render the game
            update_game_screen(screen, game_board_rect, text_area_rect, snake, food, points, high_score, episode, agent_type)

            # Control frame rate
            clock.tick(FPS)

        # Agent-specific updates post-episode
        if agent_type == 'dqn':
            agent.decay_epsilon()
            if episode % TARGET_UPDATE == 0:
                agent.update_target_network()
        elif agent_type == 'tabular':
            agent.decay_epsilon()

        # Logging and plotting
        rewards_history.append(episode_reward)
        avg_reward = sum(rewards_history[-100:]) / min(len(rewards_history), 100)
        data['episodes'].append(episode)
        data['scores'].append(points)
        data['high_scores'].append(high_score)
        data['rewards'].append(episode_reward)
        data['avg_rewards'].append(avg_reward)

        draw_plot(
            screen, 
            plot_area1, 
            data, 
            ['scores', 'high_scores'], 
            point_metrics=['scores']
        )
        draw_plot(
            screen, 
            plot_area2, 
            data, 
            ['rewards', 'avg_rewards'], 
            point_metrics=['rewards']
        )

        # Print progress every 100 episodes
        if episode % 100 == 0:
            print_progress(agent_type, episode, points, high_score, avg_reward, agent if agent_type != 'random' else None)

        # Save model if applicable
        if agent_type == 'dqn' and episode % 100 == 0:
            save_model_periodically(agent, episode)

def main():
    parser = argparse.ArgumentParser(description="Snake Game with RL Agents")
    parser.add_argument(
        '--agent', type=str, choices=['dqn', 'random', 'tabular'], default='dqn',
        help='Type of agent to run/train: dqn, random, tabular'
    )
    args = parser.parse_args()

    agent_type = args.agent
    high_score_file = HIGH_SCORE_FILES[agent_type]

    # Initialize Pygame
    pg.init()
    screen = initialize_pygame_window()
    game_board_rect, text_area_rect, plot_area1, plot_area2 = define_areas()

    # Initialize and run the selected agent
    if agent_type == 'dqn':
        agent = DQNAgent()
    elif agent_type == 'random':
        agent = RandomAgent()
    elif agent_type == 'tabular':
        agent = TabularQAgent()
    else:
        print(f"Unknown agent type: {agent_type}")
        pg.quit()
        sys.exit(1)

    try:
        generic_training_loop(agent, high_score_file, screen, game_board_rect, text_area_rect, plot_area1, plot_area2, agent_type)
    except SystemExit:
        pass
    finally:
        print("Training completed!")
        pg.quit()

if __name__ == "__main__":
    main()
