import pygame, sys
import numpy as np
from config import WIDTH, HEIGHT, GRID_SIZE


class Renderer:
    """
    Handles all Pygame rendering: initializing display, drawing maze, entities, and optional heatmap.
    """

    def __init__(self):
        pygame.init()
        pygame.font.init()
        self.font = pygame.font.SysFont("Arial", 18)
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Pac-Man DQN")
        self.clock = pygame.time.Clock()
    
    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def render(self, env, heatmap=None, stats=None):
        """
        Draw the current frame:
          - Clear screen
          - Draw walls, dots, Pac-Man, enemies
          - Optionally draw a heatmap overlay
          - Update display and tick clock
        Args:
          env: Environment instance with walls, dots, pacman_pos, enemies
          heatmap: 2D numpy array same size as grid for visualization
        """
        self._handle_events()
        
        # Clear background
        self.screen.fill((0, 0, 0))

        # Draw static and dynamic elements
        self.draw_walls(env.walls)
        self.draw_dots(env.dots)
        self.draw_pacman(env.pacman_pos)
        self.draw_enemies(env.enemies)

        # Overlay heatmap if provided
        if heatmap is not None:
            self.draw_heatmap(heatmap)
        
        if stats:
            text = f"E:{stats['episode']}  R:{stats['reward']:.2f}  ε:{stats['epsilon']:.2f}"
            surf = self.font.render(text, True, pygame.Color("white"))
            # position it at top-left with a small margin
            self.screen.blit(surf, (10, 10))

        # Flip and tick
        pygame.display.flip()
        self.clock.tick(500)

    def draw_walls(self, walls):
        """
        Draw all wall rectangles in blue.
        """
        for w in walls:
            pygame.draw.rect(self.screen, pygame.Color('blue'), w)

    def draw_dots(self, dots):
        """
        Draw small white dots at each dot location.
        """
        radius = GRID_SIZE // 6
        for x, y in dots:
            center = (x + GRID_SIZE // 2, y + GRID_SIZE // 2)
            pygame.draw.circle(self.screen, pygame.Color('white'), center, radius)

    def draw_pacman(self, pacman_pos):
        """
        Draw Pac-Man as a yellow circle.
        """
        x, y = pacman_pos
        center = (x + GRID_SIZE // 2, y + GRID_SIZE // 2)
        radius = GRID_SIZE // 2 - 2
        pygame.draw.circle(self.screen, pygame.Color('yellow'), center, radius)

    def draw_enemies(self, enemies):
        """
        Draw each enemy as a red circle.
        """
        radius = GRID_SIZE // 2 - 2
        for ex, ey, _, _ in enemies:
            center = (ex + GRID_SIZE // 2, ey + GRID_SIZE // 2)
            pygame.draw.circle(self.screen, pygame.Color('red'), center, radius)

    def draw_heatmap(self, heatmap):
        """
        Overlay a semi-transparent red heatmap.
        heatmap: 2D numpy array of shape (grid_width, grid_height)
        """
        grid_w = WIDTH // GRID_SIZE
        grid_h = HEIGHT // GRID_SIZE

        # Normalize heatmap values to 0-255
        hm = (heatmap - np.min(heatmap)) / (np.ptp(heatmap) + 1e-6)
        hm = (hm * 255).astype('uint8')

        # Create transparent surface
        surface = pygame.Surface((WIDTH, HEIGHT), flags=pygame.SRCALPHA)
        for i in range(grid_w):
            for j in range(grid_h):
                alpha = hm[i, j] // 2  # half opacity
                color = (255, 0, 0, int(alpha))
                rect = pygame.Rect(i * GRID_SIZE, j * GRID_SIZE, GRID_SIZE, GRID_SIZE)
                pygame.draw.rect(surface, color, rect)

        # Blit overlay
        self.screen.blit(surface, (0, 0))
