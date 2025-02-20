import pygame
import random
import numpy as np
import torch

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 500, 500  # Screen width and height
GRID_SIZE = 20  # Size of each grid square
PACMAN_RADIUS = GRID_SIZE // 2 - 2  # Radius of Pac-Man
ENEMY_RADIUS = GRID_SIZE // 2 - 2  # Radius of enemies
FPS = 60  # Frames per second
ENEMY_MOVE_DELAY = 500

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
for _ in range(20):  # Now 3 enemies
    x, y = get_valid_spawn()
    dx, dy = random.choice([-GRID_SIZE, GRID_SIZE]), random.choice([-GRID_SIZE, GRID_SIZE])
    enemies.append([x, y, dx, dy])


# Track time for enemy movement
last_enemy_move_time = pygame.time.get_ticks()

def get_game_state():
    """Returns a representation of the current game state as a NumPy array."""
    state = np.zeros((WIDTH // GRID_SIZE, HEIGHT // GRID_SIZE))  # Grid representation
    
    # Mark walls as -1
    for wall in walls:
        state[wall.x // GRID_SIZE, wall.y // GRID_SIZE] = -1

    # Mark Pac-Man as 1
    state[pacman_x // GRID_SIZE, pacman_y // GRID_SIZE] = 1

    # Mark enemies as -2
    for enemy in enemies:
        state[enemy[0] // GRID_SIZE, enemy[1] // GRID_SIZE] = -2

    # Mark dots as 0.5
    for dot in dots:
        state[dot[0] // GRID_SIZE, dot[1] // GRID_SIZE] = 0.5

    return torch.tensor(state, dtype=torch.float32)  # Convert to PyTorch tensor



# Main game loop
running = True
while running:
    clock.tick(FPS)  # Limit the framerate to 60 FPS
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:  # Check if user closes the game
            running = False
        elif event.type == pygame.KEYDOWN:  # Check for key press events
            new_x, new_y = pacman_x, pacman_y  # Store potential new position
            if event.key == pygame.K_LEFT:
                new_x -= pacman_speed  # Move left
            if event.key == pygame.K_RIGHT:
                new_x += pacman_speed  # Move right
            if event.key == pygame.K_UP:
                new_y -= pacman_speed  # Move up
            if event.key == pygame.K_DOWN:
                new_y += pacman_speed  # Move down
            
            # Check for collisions with walls
            new_rect = pygame.Rect(new_x, new_y, GRID_SIZE, GRID_SIZE)
            if not any(new_rect.colliderect(wall) for wall in walls):
                pacman_x, pacman_y = new_x, new_y  # Update position if no collision
    
    # Move enemies every 0.5 sec
    current_time = pygame.time.get_ticks()
    if current_time - last_enemy_move_time > ENEMY_MOVE_DELAY:
        last_enemy_move_time = current_time
        for enemy in enemies:
            moved = False  # Track if the enemy successfully moves

            # Attempt movement in multiple random directions until one works
            for _ in range(4):  # Try up to 4 different random directions
                dx, dy = random.choice([-GRID_SIZE, 0, GRID_SIZE]), random.choice([-GRID_SIZE, 0, GRID_SIZE])
                
                # Prevent standing still
                if dx == 0 and dy == 0:
                    continue
                
                new_x = enemy[0] + dx
                new_y = enemy[1] + dy
                new_rect = pygame.Rect(new_x, new_y, GRID_SIZE, GRID_SIZE)

                # Move only if there's no collision
                if not any(new_rect.colliderect(wall) for wall in walls):
                    enemy[0], enemy[1] = new_x, new_y
                    enemy[2], enemy[3] = dx, dy  # Update direction
                    moved = True
                    break  # Stop searching once a valid move is found

            # If all 4 attempts fail, the enemy keeps its previous direction but reverses it
            if not moved:
                enemy[2], enemy[3] = -enemy[2], -enemy[3]  # Reverse direction
    
    # Remove collected dots & update score
    if (pacman_x, pacman_y) in dots:
        dots.remove((pacman_x, pacman_y))
        score += 10  # Increase score
    
    # # Check if Pac-Man collides with an enemy (Game Over)
    # for enemy in enemies:
    #     if pacman_x == enemy[0] and pacman_y == enemy[1]:
    #         print("Game Over! Final Score:", score)
    #         running = False

    # Check if Pac-Man collides with an enemy (Game Over)
    for enemy in enemies:
        if pacman_x == enemy[0] and pacman_y == enemy[1]:
            # Display Game Over text
            screen.fill(BLACK)
            game_over_text = font.render("Game Over!", True, RED)
            score_text = font.render(f"Final Score: {score}", True, WHITE)
            
            screen.blit(game_over_text, (WIDTH // 2 - 60, HEIGHT // 2 - 30))
            screen.blit(score_text, (WIDTH // 2 - 80, HEIGHT // 2 + 10))
            pygame.display.flip()

            # Pause and wait for key press to quit
            pygame.time.delay(2000)  # Wait for 2 seconds
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                        waiting = False  # Exit loop

            running = False  # End the game loop
    
    # Draw everything
    screen.fill(BLACK)  # Fill background with black
    
    # Draw walls
    for wall in walls:
        pygame.draw.rect(screen, BLUE, wall)
    
    # Draw dots
    for x, y in dots:
        pygame.draw.circle(screen, WHITE, (x + GRID_SIZE // 2, y + GRID_SIZE // 2), 3)  # Small white dots
    
    # Draw Pac-Man
    pygame.draw.circle(screen, YELLOW, (pacman_x + PACMAN_RADIUS, pacman_y + PACMAN_RADIUS), PACMAN_RADIUS)
    
    # Draw enemies
    for enemy in enemies:
        pygame.draw.circle(screen, RED, (enemy[0] + ENEMY_RADIUS, enemy[1] + ENEMY_RADIUS), ENEMY_RADIUS)
    
    # Render Score
    score_text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (10, 10))

    pygame.display.flip()  # Update display

pygame.quit()  # Exit Pygame properly
