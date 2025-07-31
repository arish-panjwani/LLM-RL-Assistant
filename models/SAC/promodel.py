import torch
import torch.nn as nn
import numpy as np
import os

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = torch.tanh(self.mean(x))  # limit range to [-1, 1]
        return mean  # we only use mean for deterministic actions

class SACAgent:
    def __init__(self, state_dim, action_dim, device):
        self.device = device
        self.actor = ActorNetwork(state_dim, action_dim).to(device)

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()
        return action

    def load(self, path):
        self.actor.load_state_dict(torch.load(path, map_location=self.device))
        self.actor.eval()