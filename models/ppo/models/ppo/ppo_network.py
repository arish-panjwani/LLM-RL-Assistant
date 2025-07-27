import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ActorNetwork(nn.Module):
    """Actor network for continuous action space (prompt embeddings)"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(ActorNetwork, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Actions bounded between -1 and 1
        )
        
        # Separate network for action standard deviation
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning action mean and std
        """
        action_mean = self.actor(state)
        action_std = torch.exp(self.log_std.clamp(-20, 2))  # Clamp for numerical stability
        
        return action_mean, action_std
    
    def get_action_and_log_prob(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action and return log probability
        """
        action_mean, action_std = self.forward(state)
        
        # Create normal distribution
        dist = Normal(action_mean, action_std)
        
        # Sample action
        action = dist.sample()
        
        # Calculate log probability
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def get_log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Get log probability of given action
        """
        action_mean, action_std = self.forward(state)
        dist = Normal(action_mean, action_std)
        
        return dist.log_prob(action).sum(dim=-1, keepdim=True)

class CriticNetwork(nn.Module):
    """Critic network for value function estimation"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super(CriticNetwork, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning state value
        """
        return self.critic(state)

class PPOMemory:
    """Memory buffer for PPO training"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
    def store(self, state, action, log_prob, reward, value, done):
        """Store transition"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        
    def clear(self):
        """Clear memory"""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        
    def get_tensors(self, device):
        """Convert stored data to tensors"""
        states = torch.FloatTensor(self.states).to(device)
        actions = torch.FloatTensor(self.actions).to(device)
        log_probs = torch.FloatTensor(self.log_probs).to(device)
        rewards = torch.FloatTensor(self.rewards).to(device)
        values = torch.FloatTensor(self.values).to(device)
        dones = torch.BoolTensor(self.dones).to(device)
        
        return states, actions, log_probs, rewards, values, dones

class CustomPPO:
    """Custom PPO implementation for prompt optimization"""
    
    def __init__(self, state_dim: int, action_dim: int, lr_actor: float = 3e-4, 
                 lr_critic: float = 1e-3, gamma: float = 0.99, eps_clip: float = 0.2,
                 K_epochs: int = 4, entropy_coef: float = 0.01, device: str = 'auto'):
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using device: {self.device}")
        
        # Hyperparameters
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_coef = entropy_coef
        
        # Networks
        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(state_dim).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Memory
        self.memory = PPOMemory()
        
        # Loss function for critic
        self.mse_loss = nn.MSELoss()
        
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        """
        Select action given state
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if deterministic:
                action_mean, _ = self.actor(state_tensor)
                action = action_mean
                log_prob = torch.tensor(0.0)
            else:
                action, log_prob = self.actor.get_action_and_log_prob(state_tensor)
            
            value = self.critic(state_tensor)
        
        return action.cpu().numpy().flatten(), log_prob.cpu().item(), value.cpu().item()
    
    def store_transition(self, state, action, log_prob, reward, value, done):
        """Store transition in memory"""
        self.memory.store(state, action, log_prob, reward, value, done)
    
    def calculate_advantages(self, rewards, values, dones, next_value=0):
        """Calculate GAE advantages"""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[i]
                next_value_calc = next_value
            else:
                next_non_terminal = 1.0 - dones[i]
                next_value_calc = values[i + 1]
            
            delta = rewards[i] + self.gamma * next_value_calc * next_non_terminal - values[i]
            gae = delta + self.gamma * 0.95 * next_non_terminal * gae  # lambda = 0.95
            advantages.insert(0, gae)
        
        return torch.FloatTensor(advantages).to(self.device)
    
    def update(self):
        """Update networks using collected transitions"""
        if len(self.memory.states) == 0:
            return
        
        # Get data from memory
        states, actions, old_log_probs, rewards, values, dones = self.memory.get_tensors(self.device)
        
        # Calculate advantages and returns
        advantages = self.calculate_advantages(rewards.cpu().numpy(), values.cpu().numpy(), dones.cpu().numpy())
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(self.K_epochs):
            # Get current policy values
            new_log_probs = self.actor.get_log_prob(states, actions)
            new_values = self.critic(states).squeeze()
            
            # Calculate ratio for clipping
            ratio = torch.exp(new_log_probs.squeeze() - old_log_probs.squeeze())
            
            # Calculate surrogate losses
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Actor loss with entropy bonus
            action_mean, action_std = self.actor(states)
            entropy = Normal(action_mean, action_std).entropy().sum(dim=-1).mean()
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
            
            # Critic loss
            critic_loss = self.mse_loss(new_values, returns)
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
            
            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
        
        # Clear memory
        self.memory.clear()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item()
        }
    
    def save(self, filename: str):
        """Save model"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filename)
        
    def load(self, filename: str):
        """Load model"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
    def get_value(self, state: np.ndarray) -> float:
        """Get state value for given state"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value = self.critic(state_tensor)
        return value.cpu().item()