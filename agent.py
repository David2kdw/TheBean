import torch
import pickle
import numpy as np
from learning import DQN, ReplayMemory, train_dqn, select_action
from config import (
    HIDDEN_SIZE,
    OUTPUT_SIZE,
    MEMORY_CAPACITY,
    MEMORY_PATH,
    LR,
    GAMMA,
    BATCH_SIZE,
    EPSILON_START,
    EPSILON_DECAY,
    EPSILON_MIN,
    GRID_SIZE
)


class Agent:
    """
    DQN-based agent that manages policy and target networks, replay memory,
    epsilon-greedy action selection, and learning updates.
    """

    def __init__(self, input_dim, output_dim=OUTPUT_SIZE):
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.policy_net = DQN(input_dim, HIDDEN_SIZE, output_dim).to(self.device)
        self.target_net = DQN(input_dim, HIDDEN_SIZE, output_dim).to(self.device)
        self.update_target()

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=LR, weight_decay=1e-5
        )

        # Replay memory
        self.memory = ReplayMemory(MEMORY_CAPACITY)

        # Epsilon-greedy parameters
        self.epsilon = EPSILON_START
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_min = EPSILON_MIN
        self.steps_done = 0

    def update_target(self):
        """
        Copy policy network weights to the target network.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def store_transition(self, *args):
        """
        Save a transition (state, action, reward, next_state, done) into replay memory.
        """
        self.memory.push(*args)

    def select_action(self, state):
        """
        Choose an action using epsilon-greedy policy.

        Args:
            state (torch.Tensor): current state tensor of shape [1, state_dim]
        Returns:
            int: chosen action index
        """
        eps = self.epsilon
        action = select_action(self.policy_net, state.to(self.device), eps)
        self.steps_done += 1
        return action

    def optimize_model(self):
        """
        Perform a single DQN training step using replay memory.
        """
        train_dqn(
            self.policy_net,
            self.target_net,
            self.memory,
            self.optimizer,
            BATCH_SIZE,
            GAMMA
        )

    def decay_epsilon(self):
        """
        Decay epsilon after each episode to reduce exploration over time.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, model_path: str, memory_path: str):
        """
        Save the policy network weights to `model_path` and
        overwrite the memory file at `memory_path`.
        """
        # 1) Model
        torch.save(self.policy_net.state_dict(), model_path)

        # 2) Memory (always the same file, so it gets replaced)
        with open(memory_path, "wb") as f:
            pickle.dump(self.memory, f)

    def load(self, path: str):
        """
        Load policy network weights from disk and update target network.

        Args:
            path (str): filepath to load the model from
        """
        self.policy_net.load_state_dict(torch.load(path))
        self.update_target()

    @torch.no_grad()
    def compute_heatmap(self, env):
        gw = env.width  // GRID_SIZE
        gh = env.height // GRID_SIZE

        # 1) gather all states
        orig = list(env.pacman_pos)
        states = []
        for i in range(gw):
            for j in range(gh):
                env.pacman_pos = [i * GRID_SIZE, j * GRID_SIZE]
                states.append(env.get_state()[0])   # get_state() is [1,feat], take [0]
        env.pacman_pos = orig

        # 2) batch them
        batch = torch.stack(states).to(self.device)    # shape [gw*gh, feat_dim]

        # 3) one network call
        qvals = self.policy_net(batch)                # [gw*gh, action_dim]
        max_q,_ = qvals.max(dim=1)                    # [gw*gh]

        # 4) reshape back to grid
        heatmap = max_q.cpu().numpy().reshape(gw, gh)
        return heatmap
