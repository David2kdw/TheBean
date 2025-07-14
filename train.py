import os
import time
import json
import torch
from config import (
    NUM_EPISODES,
    TARGET_UPDATE_FREQ,
    CHECKPOINT_DIR,
    MEMORY_PATH,
    MAX_STEPS_PER_EPISODE,
    MAX_EPISODE_TIME
)
from environment import Environment
from renderer import Renderer
from agent import Agent

# File paths for checkpoints and metadata
LATEST_MODEL = os.path.join(CHECKPOINT_DIR, 'latest_model.pth')
LATEST_META  = os.path.join(CHECKPOINT_DIR, 'latest_meta.json')
FULL_TEMPLATE = os.path.join(CHECKPOINT_DIR, 'policy_ep{ep}.pth')


def main():
    """
    Main training loop with resume and checkpointing:
      - Resume from latest_model + metadata if present
      - Run episodes with optional step/time limits
      - Save rolling latest_model + memory + metadata each episode
      - Save full snapshots every 100 episodes
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Initialize environment and agent
    env = Environment()
    init_state = env.reset()
    _, state_dim = init_state.shape
    agent = Agent(input_dim=state_dim)

    # Resume from latest checkpoint if available
    last_ep = 0
    if os.path.exists(LATEST_MODEL) and os.path.exists(LATEST_META):
        with open(LATEST_META, 'r') as f:
            meta = json.load(f)
        last_ep = meta.get('episode', 0)
        agent.epsilon = meta.get('epsilon', agent.epsilon)
        agent.load(LATEST_MODEL)
        print(f"Resumed training from episode {last_ep}, epsilon={agent.epsilon:.3f}")

    start_ep = last_ep + 1

    # Training loop
    for ep in range(start_ep, NUM_EPISODES + 1):
        state = env.reset()
        total_reward = 0.0
        done = False
        step = 0
        start_time = time.time()

        renderer = Renderer()
        renderer.clock.tick(500)

        # Loop until terminal, step limit, or time limit
        while (not done
               and step < MAX_STEPS_PER_EPISODE
               and (time.time() - start_time) < MAX_EPISODE_TIME):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.optimize_model()
            state = next_state
            total_reward += reward
            step += 1
            renderer.render(env, 
                heatmap=None,
                stats={
                'episode':  ep,
                'reward':   total_reward,
                'epsilon':  agent.epsilon
                }
            )

        # If episode didn't finish by goal, note the reason
        if not done:
            reason = ('step limit reached'
                      if step >= MAX_STEPS_PER_EPISODE
                      else 'time limit reached')
            print(f"→ Episode {ep} ended early ({reason})")
        

        # Episode-end updates
        agent.decay_epsilon()
        if ep % TARGET_UPDATE_FREQ == 0:
            agent.update_target()

        print(f"Episode {ep}/{NUM_EPISODES}"
              f" | Steps: {step}"
              f" | Reward: {total_reward:.2f}"
              f" | Epsilon: {agent.epsilon:.3f}")

        # Save latest model, memory, and metadata
        agent.save(LATEST_MODEL, MEMORY_PATH)
        with open(LATEST_META, 'w') as meta_file:
            json.dump({'episode': ep, 'epsilon': agent.epsilon}, meta_file)

        # Full snapshot every 100 episodes
        if ep % 100 == 0:
            agent.save(FULL_TEMPLATE.format(ep=ep), MEMORY_PATH)
        

    print("Training complete.")


if __name__ == '__main__':
    main()
