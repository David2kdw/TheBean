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
# INPUT_SIZE = (WIDTH // GRID_SIZE) * (HEIGHT // GRID_SIZE) + 4  # +4 for direction
HIDDEN_SIZE = 256
OUTPUT_SIZE = 4  # [Left, Right, Up, Down]
MEMORY_CAPACITY = 10000

# DQN Model / Optimizer Hyperparams
LEARNING_RATE = 0.0001
BATCH_SIZE = 80
GAMMA = 0.7

# Exploration Scheduling
EPSILON_START = 1.0
EPSILON_DECAY = 0.9999
EPSILON_MIN = 0.3

# Episodes
NUM_EPISODES = 100000

############################
#          GLOBAL VARS
############################
pacman_x = GRID_SIZE * 2
pacman_y = GRID_SIZE * 2
pacman_dx, pacman_dy = 0, 0
score = 0
last_enemy_move_time = pygame.time.get_ticks()

os.makedirs("models", exist_ok=True)

MAZE = """
############################
#............##............#
#.####.#####.##.#####.####.#
#.####.#####.##.#####.####.#
#.####.#####.##.#####.####.#
#..........................#
#.####.##.########.##.####.#
#......##....##....##......#
######.#####.##.#####.##.###
######.#          #......###
######.# ######## #.####.###
######.  ########  .####.###
######.# ######## #.####.###
######.#          #.####.###
######.# ######## #.####.###
#............##............#
#.####.#####.##.#####.####.#
#.####.#####.##.#####.####.#
#.####.#####.##.#####.####.#
#..........................#
############################
""".strip("\n")

def build_maze(layout: str, grid_size: int):
    walls, dots = [], []
    rows = layout.splitlines()
    h, w = len(rows), len(rows[0])
    for j, row in enumerate(rows):
        for i, ch in enumerate(row):
            x, y = i * grid_size, j * grid_size
            if ch == "#":                    # wall tile
                walls.append(pygame.Rect(x, y, grid_size, grid_size))
            elif ch == ".":                  # corridor *with* dot
                dots.append((x, y))
            # (space) ⇒ empty corridor, no dot
    return walls, dots, w * grid_size, h * grid_size



walls, dots, WIDTH, HEIGHT = build_maze(MAZE, GRID_SIZE)
INPUT_SIZE = (WIDTH // GRID_SIZE) * (HEIGHT // GRID_SIZE) + 4  # +4 for direction

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

#############################

def get_valid_spawn():
    """Find a random position not inside a wall or Pac-Man."""
    while True:
        x = random.randint(1, (WIDTH // GRID_SIZE) - 2) * GRID_SIZE
        y = random.randint(1, (HEIGHT // GRID_SIZE) - 2) * GRID_SIZE
        new_rect = pygame.Rect(x, y, GRID_SIZE, GRID_SIZE)
        if not any(new_rect.colliderect(w) for w in walls) and (x, y) != (pacman_x, pacman_y):
            return x, y

def get_valid_spawn_player():
    """Find a random position."""
    while True:
        x = random.randint(1, (WIDTH // GRID_SIZE) - 2) * GRID_SIZE
        y = random.randint(1, (HEIGHT // GRID_SIZE) - 2) * GRID_SIZE
        new_rect = pygame.Rect(x, y, GRID_SIZE, GRID_SIZE)
        if not any(new_rect.colliderect(w) for w in walls):
            return x, y

############################
#      ENEMY SPAWN
############################
enemies = []
for _ in range(4):
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
WALL_VAL = -1.0
ENEMY_VAL = -0.8
DOT_VAL = 0.5
PACMAN_VAL = 1.0
EMPTY_VAL = 0.0

def get_game_state():
    """Tensor representation of walls, enemies, dots, Pac-Man, direction."""
    global pacman_x, pacman_y, pacman_dx, pacman_dy
    state = np.zeros((WIDTH // GRID_SIZE, HEIGHT // GRID_SIZE))

    grid_w = WIDTH // GRID_SIZE
    grid_h = HEIGHT // GRID_SIZE

    # Mark walls
    for w in walls:
        x_start = w.x // GRID_SIZE
        y_start = w.y // GRID_SIZE
        x_end = (w.x + w.width) // GRID_SIZE
        y_end = (w.y + w.height) // GRID_SIZE
        for xx in range(x_start, x_end):
            for yy in range(y_start, y_end):
                state[xx, yy] = WALL_VAL

    # Mark dots
    for dot in dots:
        dx_, dy_ = dot[0] // GRID_SIZE, dot[1] // GRID_SIZE
        if state[dx_, dy_] == 0:
            state[dx_, dy_] = DOT_VAL

    # Mark enemies
    for en in enemies:
        ex, ey = en[0] // GRID_SIZE, en[1] // GRID_SIZE
        if 0 <= ex < grid_w and 0 <= ey < grid_h:
            if state[ex, ey] in [EMPTY_VAL, DOT_VAL]:
                state[ex, ey] = ENEMY_VAL

    # Mark Pac-Man
    px, py = pacman_x // GRID_SIZE, pacman_y // GRID_SIZE
    state[px, py] = PACMAN_VAL

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
    combined = torch.cat((state_tensor, direction_tensor), dim=1)

    # print(f"state_tensor shape: {state_tensor.shape}")      # should be [1, W×H]
    # print(f"direction_tensor shape: {direction_tensor.shape}")  # should be [1, 4]
    # print(f"combined state shape: {combined.shape}")        # should be [1, W×H+4]

    return combined

############################
#   VISUALIZATION FUNCTION
############################
def visualize_game_state(state):
    flat_state = state[0].numpy()

    grid_w = WIDTH // GRID_SIZE
    grid_h = HEIGHT // GRID_SIZE
    grid_size = grid_w * grid_h

    # Correct slicing
    state_matrix = flat_state[:grid_size].reshape(grid_w, grid_h)
    direction_vector = flat_state[grid_size:]  # Last 4 values contain movement direction

    cell_width = viz_width // (WIDTH // GRID_SIZE)
    cell_height = viz_height // (HEIGHT // GRID_SIZE)

    # Draw the game state
    for x in range(WIDTH // GRID_SIZE):
        for y in range(HEIGHT // GRID_SIZE):
            val = state_matrix[x, y]
            color = BLACK

            if abs(val - WALL_VAL) < 1e-4:
                color = BLUE
            elif abs(val - DOT_VAL) < 1e-4:
                color = WHITE
            elif abs(val - ENEMY_VAL) < 1e-4:
                color = RED
            elif abs(val - PACMAN_VAL) < 1e-4:
                color = YELLOW

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
def get_dist_to_nearest(items, px, py):
    if not items:
        return 1.0
    dists = [np.linalg.norm([px - x, py - y]) for (x, y) in items]
    max_dist = np.linalg.norm([WIDTH, HEIGHT])
    return min(dists) / max_dist

def get_reward(px, py, old_x, old_y, dots, enemies, done):
    if done:
        return -2.0  # death penalty

    reward = -0.01  # small step penalty

    if (px, py) in dots:
        reward += 1.0  # reward for eating dot

    if (px, py) == (old_x, old_y):
        reward -= 0.2  # bump into wall

    # Penalty if next to any enemy (4-neighbors only)
    for ex, ey in [(e[0], e[1]) for e in enemies]:
        if abs(px - ex) + abs(py - ey) == GRID_SIZE:
            reward -= 1.5
            break

    # Distance-based shaping
    old_dot_dist = get_dist_to_nearest(dots, old_x, old_y)
    new_dot_dist = get_dist_to_nearest(dots, px, py)
    reward += 0.3 * (old_dot_dist - new_dot_dist)  # encourage getting closer to dots

    old_enemy_dist = get_dist_to_nearest([(e[0], e[1]) for e in enemies], old_x, old_y)
    new_enemy_dist = get_dist_to_nearest([(e[0], e[1]) for e in enemies], px, py)
    reward += 0.2 * (new_enemy_dist - old_enemy_dist)  # encourage moving away from enemies

    return reward



def draw_reward_heatmap():
    from matplotlib import cm
    from matplotlib.colors import Normalize

    # Get current Pac-Man location
    px, py = pacman_x, pacman_y
    old_x, old_y = px, py

    grid_w = WIDTH // GRID_SIZE
    grid_h = HEIGHT // GRID_SIZE

    heatmap = np.zeros((grid_w, grid_h))

    for gx in range(grid_w):
        for gy in range(grid_h):
            test_px, test_py = gx * GRID_SIZE, gy * GRID_SIZE
            test_rect = pygame.Rect(test_px, test_py, GRID_SIZE, GRID_SIZE)
            if any(test_rect.colliderect(w) for w in walls):
                heatmap[gx, gy] = -1.0
                continue
            r = get_reward(test_px, test_py, old_x, old_y, dots, enemies, False)
            heatmap[gx, gy] = r

    norm = Normalize(vmin=np.min(heatmap), vmax=np.max(heatmap))
    cmap = cm.get_cmap('coolwarm')

    # Clear right panel
    pygame.draw.rect(viz_screen, BLACK, (WIDTH, 0, viz_width, HEIGHT))

    cell_width = viz_width // grid_w
    cell_height = viz_height // grid_h

    for gx in range(grid_w):
        for gy in range(grid_h):
            reward_val = heatmap[gx, gy]
            if reward_val == -1.0:
                color = (0, 0, 100)  # Wall = dark blue
            else:
                rgba = cmap(norm(reward_val))
                color = tuple(int(c * 255) for c in rgba[:3])

            pygame.draw.rect(viz_screen, color, (
                WIDTH + gx * cell_width,
                gy * cell_height,
                cell_width,
                cell_height
            ))

    pygame.display.update()
    print("🔍 Reward heatmap displayed. Waiting for key...")

    # Pause until a key is pressed
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                waiting = False





############################
#   SETUP MODEL / MEMORY
############################
model = DQN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
memory = ReplayMemory(MEMORY_CAPACITY)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

target_model = DQN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
def update_target_model(online_model, target_model):
    target_model.load_state_dict(online_model.state_dict())  # Copy weights
    print("🎯 Target model updated!")


epsilon = EPSILON_START
episode_count = 0  # Keep track of episodes so we can save it
if os.path.exists("pacman_dqn.pth"):
    model.load_state_dict(torch.load("pacman_dqn.pth"))
    model.eval()
    print("Loaded existing trained model: pacman_dqn.pth")

# Load Replay Memory
if os.path.exists("replay_memory.pkl"):
    with open("replay_memory.pkl", "rb") as f:
        memory = pickle.load(f)
    print(f"Loaded existing replay memory with {len(memory)} transitions.")

# Load Epsilon & Episode Count if available
if os.path.exists("training_state.pkl"):
    with open("training_state.pkl", "rb") as f:
        saved_state = pickle.load(f)
        epsilon = saved_state.get("epsilon", EPSILON_START)
        episode_count = saved_state.get("episode_count", 0)
    print(f"Loaded epsilon={epsilon:.3f}, episode_count={episode_count} from training_state.pkl")
update_target_model(model, target_model)

############################
#       MAIN LOOP
############################
num_episodes_remaining = NUM_EPISODES - episode_count
total_move_count = 0
for episode in range(episode_count + 1, NUM_EPISODES + 1):
    # Reset Pac-Man
    pacman_x, pacman_y = GRID_SIZE * 2, GRID_SIZE * 2
    done = False
    total_reward = 0
    score = 0

    last_action_time = pygame.time.get_ticks()
    episode_start_time = pygame.time.get_ticks()

    # Randomize enemies/dots
    pacman_x, pacman_y = get_valid_spawn_player()
    randomize_enemies()
    walls, dots, WIDTH, HEIGHT = build_maze(MAZE, GRID_SIZE)

    # Initial state
    state = get_game_state()
    move_count = 0
    player_move = 0
    enemy_move = 0

    while not done:
        player_move += 1
        enemy_move += 1
        clock.tick(FPS)
        current_time = pygame.time.get_ticks()
        agent_moved = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_h:
                    draw_reward_heatmap()

        # Episode Timeout
        if current_time - episode_start_time > 30000:  # ~100s
            print(f"Episode {episode} timed out!")
            break

        # Agent Move
        if player_move % 1 == 0:
            state = get_game_state()
            move_count += 1
            total_move_count += 1
            last_action_time = current_time

            action = select_action(model, state, epsilon)

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
            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # Train every 5 steps
            if total_move_count % 10 == 0:
                train_dqn(model, target_model, memory, optimizer, BATCH_SIZE, GAMMA)
            if total_move_count % 500 == 0:
                update_target_model(model, target_model)
            if total_move_count % 2000 == 0:
                # Save Replay Memory
                with open("replay_memory.pkl", "wb") as f:
                    pickle.dump(memory, f)
                print("Replay memory saved.")

            # Dot collection
            if (pacman_x, pacman_y) in dots:
                dots.remove((pacman_x, pacman_y))
                score += 1
            agent_moved = True

        if (enemy_move % 5 == 1) and (agent_moved):
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
        score_text = font.render(f"Episode: {episode}, Reward: {total_reward:.1f}, Eps: {epsilon:.3f}", True, WHITE)
        screen.blit(score_text, (10, 10))
        visualize_game_state(state)
        pygame.display.flip()
        state = get_game_state()

    # Decay Epsilon
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    # Print Episode Info
    print(f"Episode {episode}, Score: {score}, Reward: {total_reward}, Epsilon: {epsilon:.3f}, Replay: {len(memory)}")

    # Save DQN Model
    torch.save(model.state_dict(), "pacman_dqn.pth")
    print("Model saved as pacman_dqn.pth")

    if episode % 500 == 0:
        model_path = f"models/pacman_ep{episode}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"📁 Snapshot saved: {model_path}")

    # Save Epsilon & Episode Count
    training_state = {
        "epsilon": epsilon,
        "episode_count": episode
    }
    with open("training_state.pkl", "wb") as f:
        pickle.dump(training_state, f)
    print("Training state saved.")

pygame.quit()
