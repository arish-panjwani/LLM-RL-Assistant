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
        model_path = f"{self.config.MODEL_SAVE_PATH}PPO_final.zip"
        
        if os.path.exists(model_path):
            try:
                self.model = PPO.load(model_path)
                logger.info("Loaded PPO model successfully")
            except Exception as e:
                logger.error(f"Error loading PPO model: {e}")
                self.model = None
        else:
            logger.warning(f"Model file not found: {model_path}. Model will be loaded when available.")
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
