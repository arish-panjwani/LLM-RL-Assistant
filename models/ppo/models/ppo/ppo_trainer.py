from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import os
import logging
from config.config import Config

logger = logging.getLogger(__name__)

class PPOTrainer:
    """PPO model trainer for prompt optimization"""
    
    def __init__(self, config: Config, env_fn=None):
        self.config = config
        self.env_fn = env_fn
        
        # Ensure directories exist
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.LOG_PATH, exist_ok=True)
        os.makedirs(config.TENSORBOARD_PATH, exist_ok=True)
    
    def train(self):
        """Train the PPO model"""
        logger.info("Starting PPO training...")
        
        if self.env_fn is None:
            logger.error("No environment function provided. Cannot train without environment.")
            return None
        
        # Create vectorized environment with fixed seed to avoid int32 overflow
        env = make_vec_env(self.env_fn, n_envs=self.config.N_ENVS, seed=42)
        
        # Initialize PPO model
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=f"{self.config.TENSORBOARD_PATH}PPO/",
            device="auto",
            learning_rate=3e-4,
            n_steps=64,  # Reduced from 2048 to 64 for demo
            batch_size=32,  # Reduced from 64 to 32
            n_epochs=4,  # Reduced from 10 to 4
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01
        )
        
        # Setup evaluation callback
        eval_callback = EvalCallback(
            env,
            best_model_save_path=f"{self.config.MODEL_SAVE_PATH}PPO_best/",
            log_path=f"{self.config.LOG_PATH}PPO_eval/",
            eval_freq=self.config.EVAL_FREQ,
            deterministic=True,
            render=False
        )
        
        # Train the model
        model.learn(
            total_timesteps=self.config.TOTAL_TIMESTEPS,
            callback=eval_callback,
            progress_bar=True
        )
        
        # Save the final model
        model.save(f"{self.config.MODEL_SAVE_PATH}PPO_final")
        logger.info("PPO training completed successfully")
        
        return model
