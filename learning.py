import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn

train_step = 0  # Global counter for tracking updates
losses = []  # Store loss history
q_value_logs = []  # Store Q-value history

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.1):
        super(DQN, self).__init__()
        
        # Layer 1
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act1 = nn.LeakyReLU(negative_slope=0.01)
        self.drop1 = nn.Dropout(p=dropout_rate)

        # Layer 2
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.act2 = nn.LeakyReLU(negative_slope=0.01)
        self.drop2 = nn.Dropout(p=dropout_rate)

        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.act3 = nn.LeakyReLU(negative_slope=0.01)

        self.fc4 = nn.Linear(hidden_size, hidden_size // 2)
        self.act4 = nn.LeakyReLU(negative_slope=0.01)
        # No dropout here -> keep final path a bit more “direct”
        
        # Output
        self.output_layer = nn.Linear(hidden_size // 2, output_size)
        
    def forward(self, x):
        # Flatten
        x = x.view(x.shape[0], -1)
        
        # Layer 1
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop1(x)

        # Layer 2
        x = self.fc2(x)
        x = self.act2(x)
        x = self.drop2(x)

        x = self.fc3(x)
        x = self.act3(x)

        # Layer 3
        x = self.fc4(x)
        x = self.act4(x)

        # Output
        x = self.output_layer(x)
        return x

class ReplayMemory:
    def __init__(
        self,
        capacity,
        alpha=0.7,
        min_termination_samples=20  # Minimum # of terminal samples per batch
    ):
        """
        :param capacity: Max number of transitions to store
        :param alpha:    How strongly to prioritize high-TD error samples
        :param min_termination_samples: Force at least this many terminal states per batch
        """
        self.capacity = capacity
        self.alpha = alpha
        self.min_termination_samples = min_termination_samples

        self.memory = []       # List of (state, action, reward, next_state, done)
        self.priorities = []   # List of priorities for each transition

    def push(self, state, action, reward, next_state, done):
        """
        Push a new transition into memory. We boost the priority of terminal transitions
        so they are more likely to be sampled.
        """
        # Compute a default priority = (|reward| + small_constant) ^ alpha
        # so that more “significant” (often negative) rewards get higher base priority
        base_priority = abs(reward) + 1e-5

        # Boost priority for terminal samples
        if done:
            base_priority *= 15.0  # <-- Tweak as needed

        priority = (base_priority) ** self.alpha

        # Add transition
        self.memory.append((state, action, reward, next_state, done))
        self.priorities.append(priority)

        # Remove oldest if over capacity
        if len(self.memory) > self.capacity:
            self.memory.pop(0)
            self.priorities.pop(0)

    def sample(self, batch_size, beta=0.4):
        """
        Sample a batch with prioritized replay, ensuring a minimum number
        of terminal transitions (`min_termination_samples`) are always included.
        """

        # If we don't even have enough transitions, return them all
        if len(self.memory) < batch_size:
            indices = np.arange(len(self.memory))
            samples = [self.memory[idx] for idx in indices]
            weights = np.ones(len(samples), dtype=np.float32)
            return samples, indices, torch.tensor(weights)

        # Separate indices for terminal and non-terminal transitions
        termination_indices = [i for i in range(len(self.memory)) if self.memory[i][-1]]
        non_termination_indices = [i for i in range(len(self.memory)) if not self.memory[i][-1]]

        # --------------------------
        # 1) Force Terminal Samples
        # --------------------------
        num_term_samples = min(self.min_termination_samples, len(termination_indices))
        # Choose them uniformly from terminal transitions (you could also do priority-based sampling here)
        chosen_term_indices = []
        if num_term_samples > 0:
            chosen_term_indices = np.random.choice(
                termination_indices,
                size=num_term_samples,
                replace=False
            )

        # --------------------------
        # 2) Priority Sampling for Rest
        # --------------------------
        # We only sample the *remaining* slots from the entire buffer
        # (including terminals that weren’t forced in).
        # If you prefer, you can exclude all forced-out terminal indices from priority-sampling.
        num_non_term_samples = batch_size - num_term_samples

        # Convert priorities to np.array
        priorities = np.array(self.priorities, dtype=np.float32)
        total_priority = priorities.sum()
        if total_priority <= 0.0:
            # Edge case: if all priorities are 0 (or extremely small)
            probs = np.ones(len(priorities)) / len(priorities)
        else:
            probs = priorities / total_priority

        # Sample the rest with weighted random
        chosen_other_indices = np.random.choice(
            np.arange(len(self.memory)),
            size=num_non_term_samples,
            replace=False,
            p=probs
        )

        # Combine forced terminal & priority-sampled indices
        final_indices = np.concatenate([chosen_term_indices, chosen_other_indices]).astype(np.int64)

        # Gather transitions
        samples = [self.memory[idx] for idx in final_indices]

        # --------------------------
        # 3) Compute Importance Sampling Weights
        # --------------------------
        # P(i) = p_i / sum(p), weights = (1/(N * P(i)))^beta
        chosen_probs = probs[final_indices]
        weights = (1.0 / (len(self.memory) * chosen_probs)) ** beta

        # Normalize by mean (so that max weight ~ 1)
        weights /= weights.mean()

        return samples, final_indices, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices, td_errors):
        """
        Update priorities after computing TD errors. We often use absolute TD error.
        """
        for i, td_err in zip(indices, td_errors):
            # We still do alpha exponentiation. The (5.0 boost) is handled on push
            self.priorities[i] = (abs(td_err) + 1e-5) ** self.alpha

    def __len__(self):
        return len(self.memory)


    
def train_dqn(online_model, target_model, memory, optimizer, batch_size=32, gamma=0.99, beta=0.3):
    global train_step, losses, q_value_logs

    # 1. Check if we have enough samples
    if len(memory) < batch_size:
        return

    # 2. Sample from prioritized replay memory
    batch, indices, weights = memory.sample(batch_size, beta=beta)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.cat(states, dim=0)
    next_states = torch.cat(next_states, dim=0)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.bool)
    weights = weights.to(states.device)

    # 3. Q-values of the current states (for actions taken)
    q_values = online_model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # 4. Double DQN Target Calculation
    with torch.no_grad():
        next_actions = online_model(next_states).argmax(dim=1, keepdim=True)
        next_target_q = target_model(next_states).gather(1, next_actions).squeeze(1)
        target_q_values = rewards + gamma * next_target_q * (~dones)
        target_q_values[dones] = rewards[dones]  # No future reward for terminal states


    # 5. Compute Weighted (IS) Loss
    loss_fn = nn.SmoothL1Loss(reduction='none')
    losses_raw = loss_fn(q_values, target_q_values)
    loss = (losses_raw * weights.detach()).mean()


    # 6. Optimize the network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 7. Update priorities (after backprop)
    td_errors = torch.abs(q_values - target_q_values).detach().cpu().numpy()
    memory.update_priorities(indices, td_errors)


    # 8. Logging
    losses.append(loss.item())
    q_value_logs.append(q_values.mean().item())

    train_step += 1
    if train_step % 25 == 0:
        avg_loss = np.mean(losses[-25:])
        avg_q_value = np.mean(q_value_logs[-25:])
        max_q_value = np.max(q_value_logs[-25:])
        min_q_value = np.min(q_value_logs[-25:])

        done_indices = [i for i in range(len(dones)) if dones[i]]
        not_done_indices = [i for i in range(len(dones)) if not dones[i]]
        n_terminations = len(done_indices)  # Number of termination samples

        print(f"\n🔹 TRAINING UPDATE {train_step}")
        print(f"   ✅ Average Loss (Last 25 Steps): {avg_loss:.5f}")
        print(f"   ✅ Average Q-value: {avg_q_value:.3f} (Min: {min_q_value:.3f}, Max: {max_q_value:.3f})")
        print(f"   ✅ Termination Samples in this batch: {n_terminations}")

        # Sample debug for done=False
        if not_done_indices:
            sample_idx = random.choice(not_done_indices)
            print(f"   ✅ Ongoing Sample: Q-value = {q_values[sample_idx].item():.3f} | "
                  f"Target = {target_q_values[sample_idx].item():.3f}")

        # Sample debug for done=True
        if done_indices:
            sample_idx = random.choice(done_indices)
            print(f"   ❌ Termination Sample: Q-value = {q_values[sample_idx].item():.3f} | "
                  f"Target = {target_q_values[sample_idx].item():.3f}")

        print("-" * 50)






def select_action(model, state, epsilon):
    if random.random() < epsilon:  # Exploration
        return random.randint(0, 3)  # 4 possible actions
    else:  # Exploitation
        with torch.no_grad():
            model.eval()  # Use eval mode to avoid BatchNorm issues
            q_values = model(state.unsqueeze(0))  # Add batch dimension
            model.train()  # Restore training mode
            return q_values.argmax(dim=1).item()  # Select best action


    

