# environment.py
import pygame
import random
import numpy as np
import torch
from copy import deepcopy
from config import MAZE, GRID_SIZE, WALL_VAL, DOT_VAL, ENEMY_VAL, EMPTY_VAL, PACMAN_VAL
from utils import (
    manhattan_dist,
    get_dist_to_nearest
)


class Environment:
    """
    Pac-Man environment for DQN training.
    Encapsulates maze layout, entity positions, state encoding, and reward logic.
    """

    def __init__(self):
        """
        Build the static maze and record initial dot locations.
        Randomly generate a fixed set of enemy start positions and
        set Pac-Man’s start position. Finally, call reset() to
        initialize the current episode’s state.
        """
        # Parse walls & dots, compute pixel dims
        self.walls, self.init_dots, self.width, self.height = \
            self.build_maze(MAZE, GRID_SIZE)

        # Pac-Man start (you can adjust as desired)
        self.pacman_start = (GRID_SIZE * 2, GRID_SIZE * 2)

        # Pre-pick 4 enemy spawns with random directions
        self.enemy_count = 4
        self.enemy_starts = []
        for _ in range(self.enemy_count):
            ex, ey = self._get_valid_spawn()
            dx = random.choice([-GRID_SIZE, GRID_SIZE])
            dy = random.choice([-GRID_SIZE, GRID_SIZE])
            self.enemy_starts.append((ex, ey, dx, dy))

        # Initialize current episode
        self.reset()

    def build_maze(self, maze_layout: str, grid: int):
        """
        Parse an ASCII maze into wall rectangles and dot locations.
        Returns:
          - walls: list of pygame.Rect
          - dots: list of (x, y) tuples
          - width, height: overall pixel dimensions
        """
        walls, dots = [], []
        rows = maze_layout.splitlines()
        h, w = len(rows), len(rows[0])
        for j, row in enumerate(rows):
            for i, ch in enumerate(row):
                x, y = i * grid, j * grid
                if ch == "#":
                    walls.append(pygame.Rect(x, y, grid, grid))
                elif ch == ".":
                    dots.append((x, y))
        return walls, dots, w * grid, h * grid

    def _get_valid_spawn(self):
        """
        Pick a random (x, y) not colliding with walls or Pac-Man’s start.
        """
        while True:
            gx = random.randint(1, (self.width // GRID_SIZE) - 2)
            gy = random.randint(1, (self.height // GRID_SIZE) - 2)
            x, y = gx * GRID_SIZE, gy * GRID_SIZE
            r = pygame.Rect(x, y, GRID_SIZE, GRID_SIZE)
            if not any(r.colliderect(w) for w in self.walls) and (x, y) != self.pacman_start:
                return x, y

    def reset(self):
        """
        Begin a new episode:
          1. Restore dots from the initial copy.
          2. Place Pac-Man at self.pacman_start, zero his direction.
          3. Place each enemy at its chosen start & direction.
        Returns:
          initial state tensor via get_state()
        """
        # 1. Dots
        self.dots = deepcopy(self.init_dots)

        # 2. Pac-Man
        self.pacman_pos = list(self.pacman_start)
        self.pacman_dx = 0
        self.pacman_dy = 0

        # 3. Enemies
        self.enemies = [
            [ex, ey, dx, dy]
            for (ex, ey, dx, dy) in self.enemy_starts
        ]

        return self.get_state()

    def step(self, action: int):
        """
        Apply one time-step:
          - Move Pac-Man based on action (0=←,1=→,2=↑,3=↓).
          - Check for wall collision (revert if needed).
          - Check for enemy collision → done flag.
          - Compute reward via compute_reward().
          - Remove eaten dot if present.
          - Move each enemy one random valid step.
        Returns:
          next_state (torch.Tensor), reward (float), done (bool)
        """
        old_x, old_y = self.pacman_pos.copy()

        # 1. Move Pac-Man & record direction
        if action == 0:
            self.pacman_pos[0] -= GRID_SIZE; self.pacman_dx, self.pacman_dy = -1, 0
        elif action == 1:
            self.pacman_pos[0] += GRID_SIZE; self.pacman_dx, self.pacman_dy =  1, 0
        elif action == 2:
            self.pacman_pos[1] -= GRID_SIZE; self.pacman_dx, self.pacman_dy =  0, -1
        elif action == 3:
            self.pacman_pos[1] += GRID_SIZE; self.pacman_dx, self.pacman_dy =  0,  1

        # 2. Wall collision?
        r = pygame.Rect(self.pacman_pos[0], self.pacman_pos[1], GRID_SIZE, GRID_SIZE)
        if any(r.colliderect(w) for w in self.walls):
            self.pacman_pos = [old_x, old_y]

        # 3. Enemy collision → done
        done = any((self.pacman_pos[0] == e[0] and self.pacman_pos[1] == e[1])
                   for e in self.enemies)

        # 4. Reward
        reward = self.compute_reward(old_x, old_y,
                                     self.pacman_pos[0], self.pacman_pos[1],
                                     done)

        # 5. Dot eaten?
        pos = tuple(self.pacman_pos)
        if pos in self.dots:
            self.dots.remove(pos)

        # 6. Move enemies (one random valid shift each)
        for enemy in self.enemies:
            moved = False
            for _ in range(4):
                if random.random() < 0.5:
                    dx, dy = random.choice([-GRID_SIZE, GRID_SIZE]), 0
                else:
                    dx, dy = 0, random.choice([-GRID_SIZE, GRID_SIZE])
                nx, ny = enemy[0] + dx, enemy[1] + dy
                re = pygame.Rect(nx, ny, GRID_SIZE, GRID_SIZE)
                if not any(re.colliderect(w) for w in self.walls):
                    enemy[:] = [nx, ny, dx, dy]
                    moved = True
                    break
            if not moved:
                # reverse if stuck
                enemy[2], enemy[3] = -enemy[2], -enemy[3]

        # 7. New observation
        next_state = self.get_state()
        return next_state, reward, done

    def get_state(self):
        """
        Encode the current maze + entities into a tensor:
          - A flattened grid of size (W×H) with values
            WALL_VAL, DOT_VAL, ENEMY_VAL, EMPTY_VAL, PACMAN_VAL
            :contentReference[oaicite:0]{index=0}
          - A 4-element one-hot of Pac-Man’s direction.
        Returns:
          torch.Tensor shape [1, W×H + 4]
        """
        gw, gh = self.width // GRID_SIZE, self.height // GRID_SIZE
        grid = np.zeros((gw, gh), dtype=float)

        # walls
        for w in self.walls:
            xs, ys = w.x//GRID_SIZE, w.y//GRID_SIZE
            xe, ye = (w.x+w.width)//GRID_SIZE, (w.y+w.height)//GRID_SIZE
            for x in range(xs, xe):
                for y in range(ys, ye):
                    grid[x, y] = WALL_VAL

        # dots
        for x, y in self.dots:
            gx, gy = x//GRID_SIZE, y//GRID_SIZE
            if grid[gx, gy] == EMPTY_VAL:
                grid[gx, gy] = DOT_VAL

        # enemies
        for ex, ey, _, _ in self.enemies:
            gx, gy = ex//GRID_SIZE, ey//GRID_SIZE
            if 0 <= gx < gw and 0 <= gy < gh and grid[gx, gy] in (EMPTY_VAL, DOT_VAL):
                grid[gx, gy] = ENEMY_VAL

        # Pac-Man
        px, py = self.pacman_pos[0]//GRID_SIZE, self.pacman_pos[1]//GRID_SIZE
        grid[px, py] = PACMAN_VAL

        # to tensor
        t_grid = torch.tensor(grid.flatten(), dtype=torch.float32).unsqueeze(0)

        # direction one-hot
        dir_oh = [0,0,0,0]
        if   self.pacman_dx == -1: dir_oh[0]=1
        elif self.pacman_dx ==  1: dir_oh[1]=1
        elif self.pacman_dy == -1: dir_oh[2]=1
        elif self.pacman_dy ==  1: dir_oh[3]=1
        t_dir = torch.tensor([dir_oh], dtype=torch.float32)

        return torch.cat((t_grid, t_dir), dim=1)

    def compute_reward(self, old_x, old_y, new_x, new_y, done: bool):
        """
        Shaped reward:
          - Terminal collision: -5.0
          - Living cost: -0.02
          - +1.5 for dot eaten
          - -0.4 for bumping a wall (no movement)
          - -2.0 if adjacent to an enemy
          - +0.7 * progress toward nearest dot, normalized by GRID_SIZE
          - +0.3 * progress away from nearest enemy :contentReference[oaicite:1]{index=1}
        """
        if done:
            return -5.0

        r = -0.02
        # dot
        if (new_x, new_y) in self.dots:
            r += 1.5
        # wall bump
        if (new_x, new_y) == (old_x, old_y):
            r -= 0.4
        # near enemy
        for ex, ey, _, _ in self.enemies:
            if manhattan_dist((new_x, new_y), (ex, ey)) == GRID_SIZE:
                r -= 2.0
                break
        # shaping toward dots
        old_d = get_dist_to_nearest(self.dots, old_x, old_y)
        new_d = get_dist_to_nearest(self.dots, new_x, new_y)
        r += 0.7 * ((old_d - new_d) / GRID_SIZE)
        # shaping away from enemies
        eps = [(e[0], e[1]) for e in self.enemies]
        old_e = get_dist_to_nearest(eps, old_x, old_y)
        new_e = get_dist_to_nearest(eps, new_x, new_y)
        r += 0.3 * ((new_e - old_e) / GRID_SIZE)

        return r
