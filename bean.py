import pygame
import random
import numpy as np
import torch
import learning
import os
from learning import DQN, ReplayMemory, train_dqn, select_action
import pickle

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 500, 500  # Screen width and height
GRID_SIZE = 20  # Size of each grid square
PACMAN_RADIUS = GRID_SIZE // 2 - 2  # Radius of Pac-Man
ENEMY_RADIUS = GRID_SIZE // 2 - 2  # Radius of enemies
FPS = 60  # Frames per second
ENEMY_MOVE_DELAY = 250

# Colors
BLACK = (0, 0, 0)  # Background color
YELLOW = (255, 255, 0)  # Pac-Man color
RED = (255, 0, 0)  # Enemy color
BLUE = (0, 0, 255)  # Wall color
WHITE = (255, 255, 255)  # Dot color

# Create window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simple Pac-Man")
clock = pygame.time.Clock()  # Create a clock object for controlling the framerate

# Font for score display
font = pygame.font.Font(None, 36)


# Pac-Man position (starting point)
pacman_x, pacman_y = GRID_SIZE * 2, GRID_SIZE * 2
pacman_speed = GRID_SIZE  # Movement step size
score = 0  # Player score

# Walls (list of rectangles)
walls = [
    # Outer boundary walls
    pygame.Rect(0, 0, WIDTH, GRID_SIZE),  # Top boundary
    pygame.Rect(0, HEIGHT - GRID_SIZE, WIDTH, GRID_SIZE),  # Bottom boundary
    pygame.Rect(0, 0, GRID_SIZE, HEIGHT),  # Left boundary
    pygame.Rect(WIDTH - GRID_SIZE, 0, GRID_SIZE, HEIGHT),  # Right boundary
    
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

# Dots (list of positions) excluding wall positions
dots = [(x, y) for x in range(0, WIDTH, GRID_SIZE) for y in range(0, HEIGHT, GRID_SIZE) 
        if not any(pygame.Rect(x, y, GRID_SIZE, GRID_SIZE).colliderect(wall) for wall in walls)]

def get_valid_spawn():
    """Finds a random position that is not inside a wall or the player's position."""
    while True:
        x = random.randint(1, (WIDTH // GRID_SIZE) - 2) * GRID_SIZE
        y = random.randint(1, (HEIGHT // GRID_SIZE) - 2) * GRID_SIZE
        new_rect = pygame.Rect(x, y, GRID_SIZE, GRID_SIZE)
        
        # Ensure enemy does not spawn inside a wall or on Pac-Man
        if not any(new_rect.colliderect(wall) for wall in walls) and (x, y) != (pacman_x, pacman_y):
            return x, y

# Pac-Man position (starting point)
pacman_x, pacman_y = GRID_SIZE * 2, GRID_SIZE * 2


# Initialize enemies in random positions while ensuring they are not inside walls
enemies = []
for _ in range(10):
    x, y = get_valid_spawn()
    dx, dy = random.choice([-GRID_SIZE, GRID_SIZE]), random.choice([-GRID_SIZE, GRID_SIZE])
    enemies.append([x, y, dx, dy])

def randomize_enemies():
    """Replace all enemies with new locations."""
    global enemies  # Ensure we modify the global enemies list
    num_enemies = len(enemies)  # Store current enemy count
    enemies = []  # Clear existing enemies
    for _ in range(num_enemies):  # Maintain the same number of enemies
        x, y = get_valid_spawn()
        dx, dy = random.choice([-GRID_SIZE, GRID_SIZE]), random.choice([-GRID_SIZE, GRID_SIZE])
        enemies.append([x, y, dx, dy])



# Track time for enemy movement
last_enemy_move_time = pygame.time.get_ticks()

pacman_dx, pacman_dy = 0, 0
def get_game_state():
    """Returns a tensor representation of the game state with walls, enemies, dots, and movement direction."""
    state = np.zeros((WIDTH // GRID_SIZE, HEIGHT // GRID_SIZE))

    for wall in walls:
        x_start = wall.x // GRID_SIZE
        y_start = wall.y // GRID_SIZE
        x_end = (wall.x + wall.width) // GRID_SIZE
        y_end = (wall.y + wall.height) // GRID_SIZE

        for x in range(x_start, x_end):
            for y in range(y_start, y_end):
                state[x, y] = -1

    for dot in dots:
        dx, dy = dot[0] // GRID_SIZE, dot[1] // GRID_SIZE
        if state[dx, dy] == 0:  # Only mark if it's not a wall
            state[dx, dy] = 0.5

    for enemy in enemies:
        ex, ey = enemy[0] // GRID_SIZE, enemy[1] // GRID_SIZE
        if state[ex, ey] in [0, 0.5]:  # Only mark if it's an empty or dot space
            state[ex, ey] = -2
    
    px, py = pacman_x // GRID_SIZE, pacman_y // GRID_SIZE
    state[px, py] = 5

    state_tensor = torch.tensor(state.flatten(), dtype=torch.float32).unsqueeze(0)

    pacman_direction = [0, 0, 0, 0]  # [Left, Right, Up, Down]
    if pacman_dx == -1: pacman_direction[0] = 1
    if pacman_dx == 1: pacman_direction[1] = 1
    if pacman_dy == -1: pacman_direction[2] = 1
    if pacman_dy == 1: pacman_direction[3] = 1
    direction_tensor = torch.tensor([pacman_direction], dtype=torch.float32)

    # print("Pac-Man Position:", np.count_nonzero(state == 5))
    # print("Enemy marked:", np.count_nonzero(state == -2))
    # print("Walls Marked:", np.count_nonzero(state == -1))
    # print("Dots Marked:", np.count_nonzero(state == 0.5))
    # print("empty:", np.count_nonzero(state == 0))
    return torch.cat((state_tensor, direction_tensor), dim=1)  




# Initialize the DQN model
input_size = (WIDTH // GRID_SIZE) * (HEIGHT // GRID_SIZE) + 4  # = 630
output_size = 4  # Four possible actions (left, right, up, down)
model = DQN(input_size, 300, output_size)

# Initialize Replay Memory and Optimizer
memory = ReplayMemory(10000)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training parameters
epsilon = 0.9  # Initial exploration rate
epsilon_decay = 0.998  # Decay exploration over time
min_epsilon = 0.2
gamma = 0.95  # Discount factor
batch_size = 45
num_episodes = 5000

if os.path.exists("pacman_dqn.pth"):
    model.load_state_dict(torch.load("pacman_dqn.pth"))
    model.eval()  # Set to evaluation mode
    print("Loaded existing trained model: pacman_dqn.pth")

# Load Replay Memory If Exists
memory = ReplayMemory(10000)  # Default if no file exists
if os.path.exists("replay_memory.pkl"):
    with open("replay_memory.pkl", "rb") as f:
        memory = pickle.load(f)
    print(f"Loaded existing replay memory with {len(memory)} transitions.")

def get_reward(pacman_x, pacman_y, prev_x, prev_y, dots, enemies, done):
    """
    Compute reward for the AI based on the current state.
    """
    # If the game is over, apply -5000 immediately
    if done:
        return -1000

    # If Pac-Man eats a dot, reward it
    if (pacman_x, pacman_y) in dots:
        return 40

    # If Pac-Man tries to move into a wall, small penalty
    if (pacman_x, pacman_y) == (prev_x, prev_y):
        return -20

    return -1

def visualize_game_state(state):
    """Displays the game state in a separate Pygame window."""
    viz_width = WIDTH // 2  # Half the main game size
    viz_height = HEIGHT // 2
    cell_size = viz_width // (WIDTH // GRID_SIZE)  # Adjust cell size based on grid size

    # Create a separate window
    viz_screen = pygame.display.set_mode((viz_width, viz_height))
    pygame.display.set_caption("State Visualization")

    # Process the state matrix
    state_matrix = state[0, :625].numpy().reshape(WIDTH // GRID_SIZE, HEIGHT // GRID_SIZE)

    for x in range(WIDTH // GRID_SIZE):
        for y in range(HEIGHT // GRID_SIZE):
            val = state_matrix[x, y]
            color = BLACK  # Default for empty space

            if val == -1:   # Wall
                color = BLUE
            elif val == 0.5:  # Dot
                color = WHITE
            elif val == -2:  # Enemy
                color = RED
            elif val == 5:   # Pac-Man
                color = YELLOW
            elif val == 0:
                color = BLACK

            pygame.draw.rect(viz_screen, color, (x * cell_size, y * cell_size, cell_size, cell_size))

    pygame.display.update()




# Main training loop
for episode in range(num_episodes):
    pacman_x, pacman_y = GRID_SIZE * 2, GRID_SIZE * 2  # Reset Pac-Man
    done = False
    total_reward = 0
    score = 0

    last_action_time = pygame.time.get_ticks()  # Track last action time
    episode_start_time = pygame.time.get_ticks()  # Track when the episode starts
    randomize_enemies()
    dots = [(x, y) for x in range(0, WIDTH, GRID_SIZE) for y in range(0, HEIGHT, GRID_SIZE)
            if not any(pygame.Rect(x, y, GRID_SIZE, GRID_SIZE).colliderect(wall) for wall in walls)]
    
    state = get_game_state()
    move_count = 0

    while not done:
        clock.tick(FPS)
        current_time = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        if current_time - episode_start_time > 100000:
            print(f"Episode {episode+1} timed out!")
            break


        if current_time - last_action_time > 100:
            move_count += 1
            last_action_time = current_time  # Update action time

            action = select_action(model, state, epsilon)

            prev_x, prev_y = pacman_x, pacman_y
            if action == 0: 
                pacman_x -= GRID_SIZE  # Move Left
                pacman_dx, pacman_dy = -1, 0  # Update direction
            if action == 1: 
                pacman_x += GRID_SIZE  # Move Right
                pacman_dx, pacman_dy = 1, 0
            if action == 2: 
                pacman_y -= GRID_SIZE  # Move Up
                pacman_dx, pacman_dy = 0, -1
            if action == 3: 
                pacman_y += GRID_SIZE  # Move Down
                pacman_dx, pacman_dy = 0, 1

            # Check for wall collisions
            new_rect = pygame.Rect(pacman_x, pacman_y, GRID_SIZE, GRID_SIZE)
            if any(new_rect.colliderect(wall) for wall in walls):
                pacman_x, pacman_y = prev_x, prev_y  # Undo move

            # Check game over **before** computing reward
            done = (pacman_x, pacman_y) in [(e[0], e[1]) for e in enemies]

            # Compute reward AFTER checking `done`
            reward = get_reward(pacman_x, pacman_y, prev_x, prev_y, dots, enemies, done)
            next_state = get_game_state()
            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if (move_count % 5 == 0):
                train_dqn(model, memory, optimizer, batch_size, gamma)

            # Handle bean collection
            if (pacman_x, pacman_y) in dots:
                dots.remove((pacman_x, pacman_y))
                score += 1

        # Move enemies every 0.5 sec
        if current_time - last_enemy_move_time > ENEMY_MOVE_DELAY:
            last_enemy_move_time = current_time
            for enemy in enemies:
                moved = False
                for _ in range(4):  # Try 4 directions
                    dx, dy = random.choice([-GRID_SIZE, 0, GRID_SIZE]), random.choice([-GRID_SIZE, 0, GRID_SIZE])
                    if dx == 0 and dy == 0: continue
                    
                    new_x = enemy[0] + dx
                    new_y = enemy[1] + dy
                    new_rect = pygame.Rect(new_x, new_y, GRID_SIZE, GRID_SIZE)
                    
                    if not any(new_rect.colliderect(wall) for wall in walls):
                        enemy[0], enemy[1] = new_x, new_y
                        enemy[2], enemy[3] = dx, dy
                        moved = True
                        break
                
                if not moved:
                    enemy[2], enemy[3] = -enemy[2], -enemy[3]

        # Render the game
        screen.fill(BLACK)
        for wall in walls:
            pygame.draw.rect(screen, BLUE, wall)
        for x, y in dots:
            pygame.draw.circle(screen, WHITE, (x + GRID_SIZE // 2, y + GRID_SIZE // 2), 3)
        pygame.draw.circle(screen, YELLOW, (pacman_x + PACMAN_RADIUS, pacman_y + PACMAN_RADIUS), PACMAN_RADIUS)
        for enemy in enemies:
            pygame.draw.circle(screen, RED, (enemy[0] + ENEMY_RADIUS, enemy[1] + ENEMY_RADIUS), ENEMY_RADIUS)

        # Render Score
        score_text = font.render(f"total_reward: {total_reward}", True, WHITE)
        screen.blit(score_text, (10, 10))

        # visualize_game_state(state)
        pygame.display.flip()
        

    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    # for _ in range(10):  # Train 5 times at episode end
    #     train_dqn(model, memory, optimizer, batch_size, gamma)


    print(f"Episode {episode+1}, Score: {score}, Reward: {total_reward}, Epsilon: {epsilon:.3f}, Replay: {memory.memory.__len__()}")
    torch.save(model.state_dict(), "pacman_dqn.pth")
    print("Model saved as pacman_dqn.pth")

    # Save Replay Memory
    with open("replay_memory.pkl", "wb") as f:
        pickle.dump(memory, f)
    print("Replay memory saved.")

pygame.quit()

