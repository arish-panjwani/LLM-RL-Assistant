import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, Any, List
import logging
import os

logger = logging.getLogger(__name__)

class ActorNetwork(nn.Module):
    """Actor network that outputs action probabilities for prompt modifications - LIGHTWEIGHT VERSION."""
    
    def __init__(self, input_dim: int, hidden_dims: list = [64, 32], output_dim: int = 5):
        super(ActorNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)  # Reduced dropout
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Use Sequential to match saved model format
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the actor network."""
        return F.softmax(self.network(state), dim=-1)

class CriticNetwork(nn.Module):
    """Critic network that estimates the value of the current state - LIGHTWEIGHT VERSION."""
    
    def __init__(self, input_dim: int, hidden_dims: list = [64, 32]):
        super(CriticNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)  # Reduced dropout
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        # Use Sequential to match saved model format
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the critic network."""
        return self.network(state)

class A2CModel(nn.Module):
    """Advantage Actor-Critic model for prompt optimization - LIGHTWEIGHT VERSION."""
    
    def __init__(self, 
                 state_dim: int = 30,  # Reduced from 50
                 action_dim: int = 5,  # Reduced from 10
                 hidden_dims: list = [64, 32],  # Reduced from [128, 64, 32]
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 device: str = 'cpu'):
        super(A2CModel, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = device
        
        # Initialize networks
        self.actor = ActorNetwork(state_dim, hidden_dims, action_dim).to(device)
        self.critic = CriticNetwork(state_dim, hidden_dims).to(device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Training history
        self.training_history = {
            'actor_loss': [],
            'critic_loss': [],
            'total_reward': [],
            'episode_length': []
        }
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through both actor and critic networks."""
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value
    
    def select_action(self, state: torch.Tensor, training: bool = True) -> Tuple[int, torch.Tensor]:
        """Select an action based on the current state."""
        state = state.to(self.device)
        action_probs, _ = self.forward(state)
        
        if training:
            # Sample action from probability distribution
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
        else:
            # Use epsilon-greedy for evaluation to add some randomness
            epsilon = 0.2  # 20% chance of random action
            if torch.rand(1).item() < epsilon:
                # Random action
                action = torch.randint(0, action_probs.shape[-1], (1,))
            else:
                # Greedy action
                action = torch.argmax(action_probs)
            log_prob = torch.log(action_probs[action])
            
        return action.item(), log_prob
    
    def update_model(self, 
                    states: List[np.ndarray],
                    actions: List[int],
                    rewards: List[float],
                    next_states: List[np.ndarray],
                    dones: List[bool],
                    log_probs: List[torch.Tensor]):
        """Update the model with collected transitions."""
        
        if len(states) == 0:
            return
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        next_states_tensor = torch.FloatTensor(next_states)
        dones_tensor = torch.FloatTensor(dones)
        log_probs_tensor = torch.stack(log_probs)
        
        # Update model
        self.update(states_tensor, actions_tensor, rewards_tensor,
                   next_states_tensor, dones_tensor, log_probs_tensor)
    
    def update(self, 
               states: torch.Tensor,
               actions: torch.Tensor,
               rewards: torch.Tensor,
               next_states: torch.Tensor,
               dones: torch.Tensor,
               log_probs: torch.Tensor) -> Dict[str, float]:
        """Update both actor and critic networks."""
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        log_probs = log_probs.to(self.device)
        
        # Compute current state values
        _, current_values = self.forward(states)
        current_values = current_values.squeeze()
        
        # Compute next state values
        with torch.no_grad():
            _, next_values = self.forward(next_states)
            next_values = next_values.squeeze()
            
        # Compute advantages
        advantages = rewards + self.gamma * next_values * (1 - dones) - current_values
        
        # Actor loss (policy gradient)
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Critic loss (value function)
        critic_loss = F.mse_loss(current_values, rewards + self.gamma * next_values * (1 - dones))
        
        # Update networks
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Store training history
        self.training_history['actor_loss'].append(actor_loss.item())
        self.training_history['critic_loss'].append(critic_loss.item())
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'advantage_mean': advantages.mean().item(),
            'advantage_std': advantages.std().item()
        }
    
    def save_model(self, filepath: str):
        """Save the model to a file."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'training_history': self.training_history,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'gamma': self.gamma,
                'device': self.device
            }
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load the model from a file."""
        if not os.path.exists(filepath):
            logger.warning(f"Model file not found: {filepath}")
            return
            
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Handle different model formats
        if 'actor_state_dict' in checkpoint:
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            
            if 'actor_optimizer_state_dict' in checkpoint:
                self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
                self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            
            if 'training_history' in checkpoint:
                self.training_history = checkpoint['training_history']
        else:
            # Handle minimal model format
            self.actor.load_state_dict(checkpoint)
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        if not self.training_history['actor_loss']:
            return {}
            
        return {
            'avg_actor_loss': np.mean(self.training_history['actor_loss'][-50:]),  # Reduced from 100
            'avg_critic_loss': np.mean(self.training_history['critic_loss'][-50:]),  # Reduced from 100
            'total_episodes': len(self.training_history['total_reward']),
            'avg_reward': np.mean(self.training_history['total_reward'][-50:]) if self.training_history['total_reward'] else 0,  # Reduced from 100
            'avg_episode_length': np.mean(self.training_history['episode_length'][-50:]) if self.training_history['episode_length'] else 0  # Reduced from 100
        } 