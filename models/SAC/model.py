import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = log_std.exp()
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        normal = torch.randn_like(mean)
        z = mean + std * normal
        action = torch.tanh(z)
        log_prob = (-0.5 * ((z - mean) / std).pow(2) - torch.log(std) - 0.5 * np.log(2 * np.pi)).sum(dim=1, keepdim=True)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=1, keepdim=True)
        return action, log_prob

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class SACAgent:
    def __init__(self, state_dim, action_dim, device):
        self.device = device
        self.policy = GaussianPolicy(state_dim, action_dim).to(device)
        self.q1 = QNetwork(state_dim, action_dim).to(device)
        self.q2 = QNetwork(state_dim, action_dim).to(device)
        self.target_q1 = QNetwork(state_dim, action_dim).to(device)
        self.target_q2 = QNetwork(state_dim, action_dim).to(device)

        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=3e-4)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=3e-4)

        self.replay_buffer = deque(maxlen=100000)
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        action, _ = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, float(done)))

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        transitions = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_states)
            target_q1 = self.target_q1(next_states, next_actions)
            target_q2 = self.target_q2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_value = rewards + self.gamma * (1 - dones) * target_q

        current_q1 = self.q1(states, actions)
        current_q2 = self.q2(states, actions)
        q1_loss = F.mse_loss(current_q1, target_value)
        q2_loss = F.mse_loss(current_q2, target_value)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        new_actions, log_probs = self.policy.sample(states)
        q1_new = self.q1(states, new_actions)
        q2_new = self.q2(states, new_actions)
        min_q = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha * log_probs - min_q).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        for target, param in zip(self.target_q1.parameters(), self.q1.parameters()):
            target.data.copy_(self.tau * param.data + (1 - self.tau) * target.data)
        for target, param in zip(self.target_q2.parameters(), self.q2.parameters()):
            target.data.copy_(self.tau * param.data + (1 - self.tau) * target.data)

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path))
