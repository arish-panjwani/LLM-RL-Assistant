import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Any, Optional
import logging
from utils.groq_client import GroqClient
from environment.reward_calculator import RewardCalculator
from config.config import Config

logger = logging.getLogger(__name__)

class PromptOptimizationEnv(gym.Env):
    """Environment for optimizing prompts using reinforcement learning"""
    
    def __init__(self, groq_client: GroqClient, reward_calculator: RewardCalculator, 
                 training_data: list, config: Config):
        super().__init__()
        
        self.groq_client = groq_client
        self.reward_calculator = reward_calculator
        self.training_data = training_data
        self.config = config
        
        # Define action space (continuous prompt modifications)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(config.ACTION_DIM,), dtype=np.float32
        )
        
        # Define observation space (user query embeddings)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(config.EMBEDDING_DIM,), dtype=np.float32
        )
        
        self.current_query = None
        self.current_data_point = None
        self.original_embedding = None
        self.episode_count = 0
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment and return initial observation"""
        super().reset(seed=seed)
        
        # Select random training data point
        self.current_data_point = np.random.choice(self.training_data)
        self.current_query = self.current_data_point['query']
        
        # Get embedding for current query
        self.original_embedding = self.reward_calculator.embedding_model.encode([self.current_query])[0]
        
        self.episode_count += 1
        
        return self.original_embedding.astype(np.float32), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action and return results"""
        
        # Apply action to modify prompt embedding
        modified_embedding = self.original_embedding + action * 0.1  # Scale action
        
        # Convert modified embedding back to prompt (simplified approach)
        modified_prompt = self._embedding_to_prompt(modified_embedding)
        
        # Get response from Groq
        response = self.groq_client.get_response(modified_prompt)
        
        # Calculate reward
        reward = self.reward_calculator.calculate_total_reward(
            self.current_query, modified_prompt, response
        )
        
        # Episode is done after one step
        terminated = True
        truncated = False
        
        info = {
            'original_prompt': self.current_query,
            'modified_prompt': modified_prompt,
            'response': response,
            'reward_breakdown': {
                'clarity': self.reward_calculator.calculate_clarity_reward(self.current_query, modified_prompt),
                'relevance': self.reward_calculator.calculate_relevance_reward(self.current_query, response),
                'hallucination_penalty': self.reward_calculator.calculate_hallucination_penalty(response)
            }
        }
        
        return modified_embedding.astype(np.float32), reward, terminated, truncated, info
    
    def _embedding_to_prompt(self, embedding: np.ndarray) -> str:
        """Convert embedding back to prompt (simplified approach)"""
        # This is a simplified approach - in practice, you might use more sophisticated methods
        # For now, we'll modify the original prompt based on embedding changes
        
        similarity = np.dot(embedding, self.original_embedding) / (
            np.linalg.norm(embedding) * np.linalg.norm(self.original_embedding)
        )
        
        if similarity > 0.95:
            # High similarity - minor modifications
            return f"Please provide a clear and specific answer to: {self.current_query}"
        elif similarity > 0.8:
            # Medium similarity - moderate modifications
            return f"Can you explain in detail: {self.current_query}"
        else:
            # Low similarity - major modifications
            return f"I need comprehensive information about: {self.current_query}"
