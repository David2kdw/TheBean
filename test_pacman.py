import os
import pickle
import random
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim

import learning
from learning import DQN, ReplayMemory, train_dqn, select_action

############################
#          GLOBALS
############################

# Screen / Grid Dimensions
WIDTH, HEIGHT = 500, 500
GRID_SIZE = 20

# Pac-Man / Enemy / Maze Constants
PACMAN_RADIUS = GRID_SIZE // 2 - 2
ENEMY_RADIUS = GRID_SIZE // 2 - 2
FPS = 60
ENEMY_MOVE_DELAY = 300

# Colors
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)

# RL Hyperparameters
INPUT_SIZE = (WIDTH // GRID_SIZE) * (HEIGHT // GRID_SIZE) + 4  # +4 for direction
HIDDEN_SIZE = 256
OUTPUT_SIZE = 4  # [Left, Right, Up, Down]
MEMORY_CAPACITY = 10000

# DQN Model / Optimizer Hyperparams
LEARNING_RATE = 0.0002
BATCH_SIZE = 80
GAMMA = 0.9

# Exploration Scheduling
EPSILON_START = 0.05
EPSILON_DECAY = 0.997
EPSILON_MIN = 0.2

# Episodes
NUM_EPISODES = 2000

############################
#          PYGAME INIT
############################

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simple Pac-Man")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 36)

# Visualization Window Setup
viz_width = WIDTH // 2
viz_height = HEIGHT // 2
viz_screen = pygame.display.set_mode((WIDTH + viz_width, HEIGHT))  # Extended window

############################
#          GLOBAL VARS
############################
pacman_x = GRID_SIZE * 2
pacman_y = GRID_SIZE * 2
pacman_dx, pacman_dy = 0, 0
score = 0
last_enemy_move_time = pygame.time.get_ticks()

############################
#        WALL CREATION
############################
walls = [
    # Outer boundary walls
    pygame.Rect(0, 0, WIDTH, GRID_SIZE),
    pygame.Rect(0, HEIGHT - GRID_SIZE, WIDTH, GRID_SIZE),
    pygame.Rect(0, 0, GRID_SIZE, HEIGHT),
    pygame.Rect(WIDTH - GRID_SIZE, 0, GRID_SIZE, HEIGHT),
    
    # Inner maze walls
    pygame.Rect(GRID_SIZE * 5, GRID_SIZE * 5, GRID_SIZE * 15, GRID_SIZE),
    pygame.Rect(GRID_SIZE * 5, GRID_SIZE * 20, GRID_SIZE * 15, GRID_SIZE),
    pygame.Rect(GRID_SIZE * 5, GRID_SIZE * 5, GRID_SIZE, GRID_SIZE * 5),
    pygame.Rect(GRID_SIZE * 20, GRID_SIZE * 5, GRID_SIZE, GRID_SIZE * 5),
    pygame.Rect(GRID_SIZE * 5, GRID_SIZE * 15, GRID_SIZE, GRID_SIZE * 5),
    pygame.Rect(GRID_SIZE * 20, GRID_SIZE * 15, GRID_SIZE, GRID_SIZE * 5),
    pygame.Rect(GRID_SIZE * 10, GRID_SIZE * 10, GRID_SIZE * 5, GRID_SIZE),
    pygame.Rect(GRID_SIZE * 10, GRID_SIZE * 10, GRID_SIZE, GRID_SIZE * 5),
    pygame.Rect(GRID_SIZE * 15, GRID_SIZE * 10, GRID_SIZE, GRID_SIZE * 5),
    pygame.Rect(GRID_SIZE * 15, GRID_SIZE * 15, GRID_SIZE, GRID_SIZE * 5),
    pygame.Rect(GRID_SIZE * 8, GRID_SIZE * 12, GRID_SIZE * 2, GRID_SIZE),
    pygame.Rect(GRID_SIZE * 15, GRID_SIZE * 12, GRID_SIZE * 2, GRID_SIZE),
    pygame.Rect(GRID_SIZE * 12, GRID_SIZE * 18, GRID_SIZE, GRID_SIZE * 2),
    pygame.Rect(GRID_SIZE * 12, GRID_SIZE * 8, GRID_SIZE, GRID_SIZE * 2)
]

############################
#    DOTS / SPAWN LOGIC
############################
dots = [
    (x, y) for x in range(0, WIDTH, GRID_SIZE)
    for y in range(0, HEIGHT, GRID_SIZE)
    if not any(pygame.Rect(x, y, GRID_SIZE, GRID_SIZE).colliderect(w) for w in walls)
]

def get_valid_spawn():
    """Find a random position not inside a wall or Pac-Man."""
    while True:
        x = random.randint(1, (WIDTH // GRID_SIZE) - 2) * GRID_SIZE
        y = random.randint(1, (HEIGHT // GRID_SIZE) - 2) * GRID_SIZE
        new_rect = pygame.Rect(x, y, GRID_SIZE, GRID_SIZE)
        if not any(new_rect.colliderect(w) for w in walls) and (x, y) != (pacman_x, pacman_y):
            return x, y

############################
#      ENEMY SPAWN
############################
enemies = []
for _ in range(15):
    ex, ey = get_valid_spawn()
    dx, dy = random.choice([-GRID_SIZE, GRID_SIZE]), random.choice([-GRID_SIZE, GRID_SIZE])
    enemies.append([ex, ey, dx, dy])

def randomize_enemies():
    """Regenerate enemies with new spawns."""
    global enemies
    num_enemies = len(enemies)
    enemies = []
    for _ in range(num_enemies):
        ex, ey = get_valid_spawn()
        dx, dy = random.choice([-GRID_SIZE, GRID_SIZE]), random.choice([-GRID_SIZE, GRID_SIZE])
        enemies.append([ex, ey, dx, dy])

############################
#       GAME STATE
############################
def get_game_state():
    """Tensor representation of walls, enemies, dots, Pac-Man, direction."""
    global pacman_x, pacman_y, pacman_dx, pacman_dy
    state = np.zeros((WIDTH // GRID_SIZE, HEIGHT // GRID_SIZE))

    # Mark walls
    for w in walls:
        x_start = w.x // GRID_SIZE
        y_start = w.y // GRID_SIZE
        x_end = (w.x + w.width) // GRID_SIZE
        y_end = (w.y + w.height) // GRID_SIZE
        for xx in range(x_start, x_end):
            for yy in range(y_start, y_end):
                state[xx, yy] = -1

    # Mark dots
    for dot in dots:
        dx_, dy_ = dot[0] // GRID_SIZE, dot[1] // GRID_SIZE
        if state[dx_, dy_] == 0:
            state[dx_, dy_] = 0.5

    # Mark enemies
    for en in enemies:
        ex, ey = en[0] // GRID_SIZE, en[1] // GRID_SIZE
        if state[ex, ey] in [0, 0.5]:
            state[ex, ey] = -2

    # Mark Pac-Man
    px, py = pacman_x // GRID_SIZE, pacman_y // GRID_SIZE
    state[px, py] = 5

    # Flatten
    state_tensor = torch.tensor(state.flatten(), dtype=torch.float32).unsqueeze(0)

    # Movement direction: [left, right, up, down]
    direction = [0, 0, 0, 0]
    if pacman_dx == -1: direction[0] = 1
    if pacman_dx == 1:  direction[1] = 1
    if pacman_dy == -1: direction[2] = 1
    if pacman_dy == 1:  direction[3] = 1
    direction_tensor = torch.tensor([direction], dtype=torch.float32)

    # Return combined
    return torch.cat((state_tensor, direction_tensor), dim=1)

############################
#   VISUALIZATION FUNCTION
############################
def visualize_game_state(state):
    state_matrix = state[0, :625].numpy().reshape(WIDTH // GRID_SIZE, HEIGHT // GRID_SIZE)
    direction_vector = state[0, 625:].numpy()  # Last 4 values contain movement direction

    cell_width = viz_width // (WIDTH // GRID_SIZE)
    cell_height = viz_height // (HEIGHT // GRID_SIZE)

    # Draw the game state
    for x in range(WIDTH // GRID_SIZE):
        for y in range(HEIGHT // GRID_SIZE):
            val = state_matrix[x, y]
            color = BLACK

            if val == -1:   color = BLUE   # Wall
            elif val == 0.5: color = WHITE  # Dot
            elif val == -2:  color = RED    # Enemy
            elif val == 5:   color = YELLOW  # Pac-Man

            pygame.draw.rect(viz_screen, color, (
                WIDTH + x * cell_width, 
                y * cell_height, 
                cell_width, 
                cell_height
            ))

    # Movement direction bars (draw on corresponding edges)
    bar_thickness = 8

    if direction_vector[0] == 1:  # Moving Left
        pygame.draw.rect(viz_screen, RED, (
            WIDTH, 0,  # Position at the left side
            bar_thickness, viz_height
        ))

    if direction_vector[1] == 1:  # Moving Right
        pygame.draw.rect(viz_screen, RED, (
            WIDTH + viz_width - bar_thickness, 0,  # Position at the right side
            bar_thickness, viz_height
        ))

    if direction_vector[2] == 1:  # Moving Up
        pygame.draw.rect(viz_screen, RED, (
            WIDTH, 0,  # Position at the top
            viz_width, bar_thickness
        ))

    if direction_vector[3] == 1:  # Moving Down
        pygame.draw.rect(viz_screen, RED, (
            WIDTH, viz_height - bar_thickness,  # Position at the bottom
            viz_width, bar_thickness
        ))

    pygame.display.update()

############################
#     REWARD FUNCTION
############################
def get_reward(px, py, old_x, old_y, dots, enemies, done):
    # If game is over
    if done:
        return -100
    # Dot eaten
    if (px, py) in dots:
        return 20
    # Hitting wall
    if (px, py) == (old_x, old_y):
        return -10
    # Step penalty
    return -0.5

############################
#   SETUP MODEL / MEMORY
############################
model = DQN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
memory = ReplayMemory(MEMORY_CAPACITY)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
if os.path.exists("pacman_dqn.pth"):
    model.load_state_dict(torch.load("pacman_dqn.pth"))
    model.eval()
    print("Loaded existing trained model: pacman_dqn.pth")


############################
#       MAIN LOOP
############################

for episode in range(5):
    # Reset Pac-Man
    pacman_x, pacman_y = GRID_SIZE * 2, GRID_SIZE * 2
    done = False
    total_reward = 0
    score = 0

    last_action_time = pygame.time.get_ticks()
    episode_start_time = pygame.time.get_ticks()

    # Randomize enemies/dots
    randomize_enemies()
    dots = [
        (x, y) for x in range(0, WIDTH, GRID_SIZE)
        for y in range(0, HEIGHT, GRID_SIZE)
        if not any(pygame.Rect(x, y, GRID_SIZE, GRID_SIZE).colliderect(w) for w in walls)
    ]

    # Initial state
    state = get_game_state()
    move_count = 0

    while not done:
        clock.tick(FPS)
        current_time = pygame.time.get_ticks()
        agent_moved = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # Episode Timeout
        if current_time - episode_start_time > 100000:  # ~100s
            print(f"Episode {episode} timed out!")
            break

        # Agent Move
        if current_time - last_action_time > 100:
            state = get_game_state()
            move_count += 1
            last_action_time = current_time

            action = select_action(model, state, EPSILON_START)

            old_x, old_y = pacman_x, pacman_y
            if action == 0:
                pacman_x -= GRID_SIZE
                pacman_dx, pacman_dy = -1, 0
            if action == 1:
                pacman_x += GRID_SIZE
                pacman_dx, pacman_dy = 1, 0
            if action == 2:
                pacman_y -= GRID_SIZE
                pacman_dx, pacman_dy = 0, -1
            if action == 3:
                pacman_y += GRID_SIZE
                pacman_dx, pacman_dy = 0, 1

            # Wall collision check
            new_rect = pygame.Rect(pacman_x, pacman_y, GRID_SIZE, GRID_SIZE)
            if any(new_rect.colliderect(w) for w in walls):
                pacman_x, pacman_y = old_x, old_y  # Revert move

            # Check game over
            done = (pacman_x, pacman_y) in [(e[0], e[1]) for e in enemies]

            # Reward
            reward = get_reward(pacman_x, pacman_y, old_x, old_y, dots, enemies, done)
            next_state = get_game_state()
            state = next_state
            total_reward += reward

            # Dot collection
            if (pacman_x, pacman_y) in dots:
                dots.remove((pacman_x, pacman_y))
                score += 1
            agent_moved = True

        # if (current_time - last_enemy_move_time > ENEMY_MOVE_DELAY) and (agent_moved):
        if (False) and (agent_moved):
            last_enemy_move_time = current_time
            for enemy in enemies:
                moved = False
                for _ in range(4):  # Try up to 4 times to find a valid move
                    if random.random() < 0.5:
                        dx, dy = random.choice([-GRID_SIZE, GRID_SIZE]), 0  # Horizontal move
                    else:
                        dx, dy = 0, random.choice([-GRID_SIZE, GRID_SIZE])  # Vertical move

                    new_x = enemy[0] + dx
                    new_y = enemy[1] + dy
                    new_rect = pygame.Rect(new_x, new_y, GRID_SIZE, GRID_SIZE)

                    if not any(new_rect.colliderect(w) for w in walls):
                        enemy[0], enemy[1] = new_x, new_y
                        enemy[2], enemy[3] = dx, dy
                        moved = True
                        break  # Exit loop once a valid move is found

                    if not moved:
                        enemy[2], enemy[3] = -enemy[2], -enemy[3]  # Reverse direction if stuck

        # Render Main Game
        screen.fill(BLACK)
        for w in walls:
            pygame.draw.rect(screen, BLUE, w)
        for x, y in dots:
            pygame.draw.circle(screen, WHITE, (x + PACMAN_RADIUS, y + PACMAN_RADIUS), 3)
        pygame.draw.circle(screen, YELLOW, (pacman_x + PACMAN_RADIUS, pacman_y + PACMAN_RADIUS), PACMAN_RADIUS)
        for en in enemies:
            pygame.draw.circle(screen, RED, (en[0] + ENEMY_RADIUS, en[1] + ENEMY_RADIUS), ENEMY_RADIUS)

        # Render Score
        score_text = font.render(f"Episode: {episode}, Reward: {total_reward}, Eps: {EPSILON_START:.3f}", True, WHITE)
        screen.blit(score_text, (10, 10))
        visualize_game_state(state)
        pygame.display.flip()
        state = get_game_state()


    # Print Episode Info
    print(f"Episode {episode}, Score: {score}, Reward: {total_reward}, Epsilon: {EPSILON_START:.3f}, Replay: {len(memory)}")

pygame.quit()
