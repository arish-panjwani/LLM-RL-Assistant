import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

class DDPGActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DDPGActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.out(x))

class DDPGAgent:
    def __init__(self, state_dim, action_dim, device):
        self.device = device
        self.actor = DDPGActor(state_dim, action_dim).to(device)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            return self.actor(state).cpu().numpy()

    def load(self, path="saved_model/ddpg_actor.pth"):
        self.actor.load_state_dict(torch.load(path, map_location=self.device))


    def train_step(self):
        pass  # Optional: add training logic

    def store_transition(self, state, action, reward, next_state, done):
        pass  # Optional: for replay buffer

    def save(self, path="saved_model/ddpg_actor.pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.actor.state_dict(), path)

    def load(self, path="saved_model/ddpg_actor.pth"):
        self.actor.load_state_dict(torch.load(path, map_location=self.device))