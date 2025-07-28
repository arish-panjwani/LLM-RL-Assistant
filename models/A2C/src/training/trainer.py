import torch
import numpy as np
import logging
import os
from typing import Dict, Any, List, Optional
from tqdm import tqdm

# Use absolute imports to avoid relative import issues
try:
    from models.a2c_model import A2CModel
except ImportError:
    # Fallback for when running from different context
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from models.a2c_model import A2CModel

logger = logging.getLogger(__name__)

class A2CTrainer:
    """A2C Trainer for prompt optimization - LIGHTWEIGHT VERSION."""
    
    def __init__(self, model: A2CModel, environment, config: Dict[str, Any]):
        self.model = model
        self.environment = environment
        self.config = config
        
        # Training parameters
        self.episodes = config.get('episodes', 50)
        self.max_steps = config.get('max_steps_per_episode', 10)
        self.batch_size = config.get('batch_size', 8)
        self.update_frequency = config.get('update_frequency', 5)
        self.save_frequency = config.get('save_frequency', 10)
        self.eval_frequency = config.get('eval_frequency', 10)
        
        # Training history
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'actor_losses': [],
            'critic_losses': [],
            'improvements': []
        }
    
    def train(self, save_path: str = "data/models/a2c_domain_agnostic_best.pth"):
        """Train the A2C model."""
        logger.info(f"Starting A2C training for {self.episodes} episodes")
        
        best_reward = float('-inf')
        patience_counter = 0
        early_stopping_patience = self.config.get('early_stopping_patience', 10)
        
        # Training loop
        for episode in range(self.episodes):
            # Reset environment
            state = self.environment.reset()
            episode_reward = 0.0
            episode_length = 0
            episode_improvements = []
            
            # Episode buffer
            states, actions, rewards, next_states, dones, log_probs = [], [], [], [], [], []
            
            # Run episode
            for step in range(self.max_steps):
                # Select action
                action, log_prob = self.model.select_action(torch.FloatTensor(state))
                
                # Take step in environment
                next_state, reward, done, info = self.environment.step(action)
                
                # Store transition
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)
                log_probs.append(log_prob)
                
                # Track improvements
                if 'improvement' in info:
                    episode_improvements.append(info['improvement'])
                
                episode_reward += reward
                episode_length += 1
                state = next_state
                
                if done:
                    break
            
            # Update model if we have enough transitions
            if len(states) > 0:
                self.model.update_model(states, actions, rewards, next_states, dones, log_probs)
            
            # Store episode results
            self.training_history['episode_rewards'].append(episode_reward)
            self.training_history['episode_lengths'].append(episode_length)
            if episode_improvements:
                self.training_history['improvements'].append(np.mean(episode_improvements))
            
            # Log progress
            if (episode + 1) % 5 == 0:
                avg_reward = np.mean(self.training_history['episode_rewards'][-5:])
                avg_improvement = np.mean(self.training_history['improvements'][-5:]) if self.training_history['improvements'] else 0.0
                logger.info(f"Episode {episode + 1}/{self.episodes} - "
                          f"Reward: {episode_reward:.3f}, Avg: {avg_reward:.3f}, "
                          f"Improvement: {avg_improvement:.3f}")
            
            # Save model periodically
            if (episode + 1) % self.save_frequency == 0:
                checkpoint_path = f"data/models/checkpoint_episode_{episode + 1}.pth"
                self.model.save_model(checkpoint_path)
                logger.info(f"Saved checkpoint at episode {episode + 1}")
            
            # Early stopping
            if episode_reward > best_reward:
                best_reward = episode_reward
                patience_counter = 0
                # Save best model
                self.model.save_model(save_path)
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at episode {episode + 1}")
                break
        
        logger.info(f"Training completed. Best reward: {best_reward:.3f}")
        return self.training_history
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        if not self.training_history['episode_rewards']:
            return {}
        
        return {
            'total_episodes': len(self.training_history['episode_rewards']),
            'avg_reward': np.mean(self.training_history['episode_rewards']),
            'best_reward': np.max(self.training_history['episode_rewards']),
            'avg_episode_length': np.mean(self.training_history['episode_lengths']),
            'avg_improvement': np.mean(self.training_history['improvements']) if self.training_history['improvements'] else 0.0
        } 