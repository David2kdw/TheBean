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
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size, hidden_size)  # New hidden layer
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(hidden_size, output_size)  # Output layer

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)  # Pass through the new hidden layer
        x = self.relu4(x)
        x = self.fc5(x)  # Output layer
        return x


    
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
def train_dqn(model, memory, optimizer, batch_size=32, gamma=0.99):
    global train_step, losses, q_value_logs

    if len(memory) < batch_size:
        return  # Wait until enough samples are in memory

    batch = memory.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.cat(states, dim=0)
    next_states = torch.cat(next_states, dim=0)

    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.bool)

    # Compute current Q-values
    q_values = model(states)  # Forward pass
    q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Compute target Q-values
    with torch.no_grad():
        next_q_values = model(next_states).max(1)[0]
        target_q_values = rewards + gamma * next_q_values * ~dones

    # Compute loss
    loss = nn.MSELoss()(q_values, target_q_values)

    # Store loss and Q-values for debugging
    losses.append(loss.item())
    q_value_logs.append(q_values.mean().item())

    # Gradient update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_step += 1  # Increment step count

    ### 🔹 Print Progress Every 100 Training Steps
    if train_step % 50 == 0:
        avg_loss = np.mean(losses[-100:])  # Average loss over the last 100 steps
        avg_q_value = np.mean(q_value_logs[-100:])  # Average Q-value
        max_q_value = np.max(q_value_logs[-100:])
        min_q_value = np.min(q_value_logs[-100:])

        print(f"\n🔹 TRAINING UPDATE {train_step}")
        print(f"   ✅ Average Loss (Last 100 Steps): {avg_loss:.5f}")
        print(f"   ✅ Average Q-value: {avg_q_value:.3f} (Min: {min_q_value:.3f}, Max: {max_q_value:.3f})")
        print(f"   ✅ Sample Q-value: {q_values[0].item():.3f} | Target: {target_q_values[0].item():.3f}")
        print("-" * 50)



def select_action(model, state, epsilon):
    if random.random() < epsilon:  # Explore
        return random.randint(0, 3)  # 4 possible actions
    else:  # Exploit
        with torch.no_grad():
            q_values = model(state)  # Ensure shape (1, output_size)
            return q_values.argmax(dim=1).item()  # Select best action


    

