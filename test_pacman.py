import os
import sys
import argparse
import torch
import pygame

from config import CHECKPOINT_DIR, INPUT_SIZE   # ← 引入 INPUT_SIZE（堆叠后的输入维度）
from environment import Environment
from renderer import Renderer
from agent import Agent


def main():
    parser = argparse.ArgumentParser(
        description='Run a trained Pac-Man DQN model with rendering for testing.'
    )
    parser.add_argument(
        '--model', '-m', type=str,
        help='Path to a .pth model checkpoint. If omitted, uses CHECKPOINT_DIR/latest_model.pth.'
    )
    parser.add_argument(
        '--episodes', '-e', type=int, default=5,
        help='Number of test episodes to run (default: 5).'
    )
    parser.add_argument(
        '--fps', type=int, default=30,
        help='Rendering frames per second (default: 30).'
    )
    args = parser.parse_args()

    # Determine checkpoint to load: argument or latest_model.pth
    ckpt = args.model or os.path.join(CHECKPOINT_DIR, 'latest_model.pth')
    if not ckpt or not os.path.isfile(ckpt):
        print(f"Error: No valid checkpoint found at {ckpt}")
        sys.exit(1)

    # Initialize environment, agent, renderer
    env = Environment()

    # 关键：用堆叠后的 INPUT_SIZE 初始化 Agent（不要用 env.get_state() 的单帧维度）
    agent = Agent(input_dim=INPUT_SIZE)
    agent.load(ckpt)
    agent.epsilon = 0.0  # greedy policy

    pygame.init()
    renderer = Renderer()

    print(f"Loaded checkpoint: {ckpt}")
    print(f"Running {args.episodes} test episodes...")

    for ep in range(1, args.episodes + 1):
        state = agent.reset_episode(env)   # [1, INPUT_SIZE]
        done = False
        total_reward = 0.0
        steps = 0

        renderer.render(env)

        while not done:
            # OS events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            action = agent.select_action(state)

            s_t, a, r, s_next, done = agent.step(env, action)

            state = s_next
            total_reward += r
            steps += 1

            renderer.render(env)
            renderer.clock.tick(args.fps)

        print(f"[Test {ep}] Steps: {steps}, Total Reward: {total_reward:.2f}")

    pygame.quit()


if __name__ == '__main__':
    main()
