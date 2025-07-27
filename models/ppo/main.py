# main.py

import logging
import argparse
from training.train_ppo import train_models, setup_directories, test_system, evaluate_models, run_server
from config.config import config  # use shared instance

def main():
    """Main entry point for PPO training"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting PPO Prompt Optimization Training...")

    # Create a simple args object for the training function
    class Args:
        def __init__(self):
            self.episodes = 500
            self.update_freq = 20
            self.eval_freq = 100
            self.save_freq = 200
            self.host = '0.0.0.0'
            self.port = 5000
            self.debug = False
    
    args = Args()
    
    # Setup directories
    setup_directories(config)
    
    # Test system first
    if not test_system(config):
        logger.error("System tests failed! Aborting.")
        return
    
    # Train models
    trained_model = train_models(config, args)
    
    # Evaluate models
    evaluate_models(config)
    
    # Start server
    logger.info("Training completed! Starting server...")
    run_server(config, args)

if __name__ == "__main__":
    main()