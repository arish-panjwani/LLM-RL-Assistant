import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import torch.optim as optim

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = torch.tanh(self.fc3(x))
        std = torch.exp(self.log_std)
        return mean, std

    def sample_action(self, state):
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        action = normal.sample()
        log_prob = normal.log_prob(action)
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.out(x)

class A2CBuffer:
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def push(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def get_batch(self):
        return (
            torch.stack(self.states),
            torch.stack(self.actions),
            torch.tensor(self.rewards, dtype=torch.float32),
            torch.stack(self.log_probs),
            torch.stack(self.values),
            torch.tensor(self.dones, dtype=torch.float32)
        )

class A2CAgent:
    def __init__(self, state_dim, action_dim, device):
        self.device = device
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.buffer = A2CBuffer()
        
        # A2C hyperparameters
        self.gamma = 0.99
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5

    def select_action(self, state):
        with torch.no_grad():
            action, log_prob = self.actor.sample_action(state.to(self.device))
            value = self.critic(state.to(self.device))
            return action.cpu().numpy()[0], log_prob.cpu(), value.cpu()

    def compute_returns(self, rewards, values, dones):
        """Compute returns using TD(0) for A2C"""
        returns = []
        next_value = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]
            
            # TD(0) return calculation
            td_return = rewards[i] + self.gamma * next_value * (1 - dones[i])
            returns.insert(0, td_return)
        
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = returns - torch.tensor(values, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def train_step(self, state, action, reward, log_prob, value, done):
        self.buffer.push(state.cpu(), action.cpu(), reward, log_prob.cpu(), value.cpu(), done)
        
        # Only train when buffer is full enough
        if len(self.buffer.states) < 64:
            return

        states, actions, rewards, log_probs, values, dones = self.buffer.get_batch()
        states = states.to(self.device)
        actions = actions.to(self.device)
        log_probs = log_probs.to(self.device)
        values = values.to(self.device)

        # Compute advantages and returns
        advantages, returns = self.compute_returns(rewards, values.cpu().numpy(), dones.cpu().numpy())
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        # A2C update
        # Get current policy distribution
        mean, std = self.actor(states)
        normal = torch.distributions.Normal(mean, std)
        new_log_probs = normal.log_prob(actions)
        
        # Actor loss (policy gradient)
        actor_loss = -(new_log_probs * advantages).mean()
        
        # Critic loss (value function)
        new_values = self.critic(states)
        critic_loss = F.mse_loss(new_values.squeeze(), returns)
        
        # Entropy bonus for exploration
        entropy = normal.entropy().mean()
        
        # Total loss
        total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
        
        # Update networks
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        # Clear buffer after training
        self.buffer.clear()

    def save(self, path):
        torch.save(self.actor.state_dict(), path)

    def load(self, path):
        self.actor.load_state_dict(torch.load(path)) 