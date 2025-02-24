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

        # Smaller third layer
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.act3 = nn.LeakyReLU(negative_slope=0.01)
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

        # Layer 3
        x = self.fc3(x)
        x = self.act3(x)

        # Output
        x = self.output_layer(x)
        return x



    
class ReplayMemory:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.memory = []
        self.alpha = alpha  # How much prioritization we use
        self.priorities = []  # Stores TD-errors

    def push(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities, default=1.0)  # New samples start with max priority
        self.memory.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)

        if len(self.memory) > self.capacity:
            del self.memory[0]
            del self.priorities[0]

    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities) ** self.alpha
        probs = priorities / priorities.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]

        # Importance sampling weights to correct bias
        weights = (len(self.memory) * probs[indices]) ** -beta
        weights /= weights.max()

        return samples, indices, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices, td_errors):
        for i, td_err in zip(indices, td_errors):
            self.priorities[i] = abs(td_err) + 1e-5  # Avoid zero priority

    def __len__(self):
        """Return the current size of the replay memory."""
        return len(self.memory)
    
def train_dqn(online_model, target_model, memory, optimizer, batch_size=32, gamma=0.99, beta=0.4):
    global train_step, losses, q_value_logs

    if len(memory) < batch_size:
        return

    # Sample from prioritized replay memory
    batch, indices, weights = memory.sample(batch_size, beta=beta)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.cat(states, dim=0)
    next_states = torch.cat(next_states, dim=0)

    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.bool)
    weights = weights.to(states.device)

    # Compute Q-values of the current states
    q_values = online_model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_online_q = online_model(next_states)
        next_actions = next_online_q.argmax(dim=1, keepdim=True)
        next_target_q = target_model(next_states).gather(1, next_actions).squeeze(1)
        target_q_values = rewards + gamma * next_target_q * ~dones

    # Compute loss (weighted for Prioritized Replay)
    loss_fn = nn.SmoothL1Loss(reduction='none')
    losses_raw = loss_fn(q_values, target_q_values)
    loss = (losses_raw * weights).mean()

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update priorities in memory
    td_errors = (q_values - target_q_values).detach().cpu().numpy()
    memory.update_priorities(indices, td_errors)

    # Logging
    losses.append(loss.item())
    q_value_logs.append(q_values.mean().item())

    train_step += 1
    if train_step % 25 == 0:
        avg_loss = np.mean(losses[-25:])
        avg_q_value = np.mean(q_value_logs[-25:])
        max_q_value = np.max(q_value_logs[-25:])
        min_q_value = np.min(q_value_logs[-25:])

        # Find indices where done=True and done=False
        done_indices = [i for i in range(len(dones)) if dones[i]]
        not_done_indices = [i for i in range(len(dones)) if not dones[i]]

        print(f"\n🔹 TRAINING UPDATE {train_step}")
        print(f"   ✅ Average Loss (Last 25 Steps): {avg_loss:.5f}")
        print(f"   ✅ Average Q-value: {avg_q_value:.3f} (Min: {min_q_value:.3f}, Max: {max_q_value:.3f})")

        # Log one sample where done=False (ongoing episode)
        if not_done_indices:
            sample_idx = random.choice(not_done_indices)
            print(f"   ✅ Ongoing Sample: Q-value = {q_values[sample_idx].item():.3f} | Target = {target_q_values[sample_idx].item():.3f}")

        # Log one sample where done=True (episode ended)
        if done_indices:
            sample_idx = random.choice(done_indices)
            print(f"   ❌ Termination Sample: Q-value = {q_values[sample_idx].item():.3f} | Target = {target_q_values[sample_idx].item():.3f}")

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


    

