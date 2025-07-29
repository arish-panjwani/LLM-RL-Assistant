from stable_baselines3 import PPO
import os
import logging
from config.config import Config
import numpy as np

logger = logging.getLogger(__name__)

class PPOModel:
    """Wrapper for trained PPO model"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load trained PPO model"""
        # Try multiple possible model paths
        possible_paths = [
            os.path.join(self.config.MODEL_SAVE_PATH, "PPO_final.zip"),
            os.path.join(self.config.MODEL_SAVE_PATH, "PPO_best", "best_model.zip"),
            os.path.join(self.config.MODEL_SAVE_PATH, "ppo", "ppo_model.zip")
        ]
        
        for model_path in possible_paths:
            if os.path.exists(model_path):
                try:
                    self.model = PPO.load(model_path)
                    logger.info(f"Loaded PPO model successfully from: {model_path}")
                    return
                except Exception as e:
                    logger.error(f"Error loading PPO model from {model_path}: {e}")
                    continue
        
        # If no model found, log warning
        logger.warning(f"No PPO model found in any of these paths: {possible_paths}")
        self.model = None
    
    def optimize_prompt(self, embedding: np.ndarray) -> np.ndarray:
        """
        Optimize prompt embedding using PPO model
        Returns modified embedding
        """
        if self.model is None:
            logger.warning("PPO model not loaded, returning original embedding")
            return embedding
        
        action, _ = self.model.predict(embedding, deterministic=True)
        return embedding + action * 0.1  # Scale action
