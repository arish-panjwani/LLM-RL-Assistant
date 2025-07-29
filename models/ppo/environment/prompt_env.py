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
        # Use 'prompt' instead of 'query'
        self.current_query = self.current_data_point['prompt']
        
        # Get embedding for current query
        self.original_embedding = self.reward_calculator.embedding_model.encode([self.current_query])[0]
        
        self.episode_count += 1
        
        return self.original_embedding.astype(np.float32), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action and return results"""
        
        # Apply action to modify embedding
        modified_embedding = self.original_embedding + action * 0.1

        # Get LLM-optimized prompt
        modified_prompt = self._embedding_to_prompt(modified_embedding)
        
        # Get response for the optimized prompt
        response = self.groq_client.get_response(modified_prompt)
        
        # Calculate meta-reward for prompt optimization quality
        meta_reward = self.reward_calculator.calculate_total_reward(
            self.current_query,
            modified_prompt,
            response,
            context=self.current_data_point.get('context'),
            depth=self.current_data_point.get('depth')
        )
        
        # Episode ends after one step
        terminated = True
        truncated = False
        
        info = {
            'original_prompt': self.current_query,
            'modified_prompt': modified_prompt,
            'response': response,
            'meta_prompt_used': True,
            'context': self.current_data_point.get('context'),
            'depth': self.current_data_point.get('depth')
        }
        
        return modified_embedding.astype(np.float32), meta_reward, terminated, truncated, info
    
    def _embedding_to_prompt(self, embedding: np.ndarray) -> str:
        """Use LLM to dynamically generate optimized prompts"""
        
        # Get context and depth
        context = self.current_data_point.get('context', 'general')
        depth = self.current_data_point.get('depth', 'intermediate')
        
        # Calculate embedding similarity for strategy selection
        similarity = np.dot(embedding, self.original_embedding) / (
            np.linalg.norm(embedding) * np.linalg.norm(self.original_embedding)
        )
        
        # Create dynamic meta-prompt based on context and depth
        meta_prompt = f"""As an expert prompt engineer, optimize this query: "{self.current_query}"

Context: {context}
Depth: {depth}
Optimization Goals:
1. Generate a prompt appropriate for {depth} level understanding
2. Focus on {context} context and applications
3. Use varied and engaging language
4. Maintain technical accuracy
5. Encourage detailed, structured responses

Requirements:
- Do NOT start with "Please provide"
- Use diverse prompt structures
- Match the complexity to the {depth} level
- Maintain focus on {context} aspects
- Encourage analytical thinking

Generate only the optimized prompt, no additional text."""

        # Get optimized prompt from LLM
        optimized_prompt = self.groq_client.get_response(meta_prompt).strip()
        
        # Validate and fallback if needed
        if not optimized_prompt or len(optimized_prompt) < 20:
            # Use context-aware fallback patterns
            patterns = {
                ('academic', 'expert'): f"Analyze the theoretical foundations and advanced implications of {self.current_query}",
                ('scientific', 'detailed'): f"Examine the mechanisms and processes involved in {self.current_query}",
                ('technical', 'expert'): f"Evaluate the architectural components and technical implementation of {self.current_query}",
                ('practical', 'beginner'): f"In simple terms, explain how {self.current_query} works and its everyday applications",
                ('current_events', 'comprehensive'): f"Analyze the current state, implications, and future trends of {self.current_query}"
            }
            
            return patterns.get((context, depth), f"Explain the key concepts and applications of {self.current_query}")
            
        return optimized_prompt