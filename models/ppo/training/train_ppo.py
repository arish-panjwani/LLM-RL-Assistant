# main.py - Main entry point for RL Prompt Optimization System
import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from config.config import Config
from utils.evaluation import ModelEvaluator
from server.deployment_server import DeploymentServer
from data.data_loader import DataLoader
from environment.reward_calculator import RewardCalculator
from models.ppo.ppo_trainer import PPOTrainer
from utils.groq_client import GroqClient
from environment.prompt_env import PromptOptimizationEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories(config: Config):
    """Create necessary directories"""
    directories = [
        config.MODEL_SAVE_PATH,
        config.LOG_PATH,
        config.TENSORBOARD_PATH,
        'data/datasets',
        'notebooks',
        'tests'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")

def train_models(config: Config, args):
    """Train RL models"""
    logger.info("Starting model training...")
    
    try:
        def create_env():
            groq_client = GroqClient(config.GROQ_API_KEY)
            reward_calculator = RewardCalculator(groq_client, config)
            data_loader = DataLoader()
            
            try:
                persona_data, hh_data, truth_data = data_loader.load_training_data()
                
                # Convert data format to match expected structure
                formatted_data = []
                
                # Process persona data with error checking and better field access
                for item in persona_data:
                    if isinstance(item, dict):
                        prompt = ""
                        # Try different possible field names
                        for field in ["prompt", "query", "text", "question"]:
                            if field in item and item[field]:
                                prompt = str(item[field])
                                break
                        
                        if prompt:  # Only add if we found a valid prompt
                            formatted_data.append({
                                "prompt": prompt,
                                "context": str(item.get("context", "general")),
                                "depth": "intermediate",
                                "type": "conversation"
                            })
                
                # Process HH data with error checking
                for item in hh_data:
                    if isinstance(item, dict):
                        prompt = ""
                        for field in ["prompt", "text", "input"]:
                            if field in item and item[field]:
                                prompt = str(item[field])
                                break
                                
                        if prompt:
                            formatted_data.append({
                                "prompt": prompt,
                                "context": "practical",
                                "depth": str(item.get("depth", "beginner")),
                                "type": "feedback"
                            })
                
                # Process truth data with error checking
                for item in truth_data:
                    if isinstance(item, dict):
                        prompt = ""
                        for field in ["prompt", "question", "input"]:
                            if field in item and item[field]:
                                prompt = str(item[field])
                                break
                                
                        if prompt:
                            formatted_data.append({
                                "prompt": prompt,
                                "context": "factual",
                                "depth": "expert",
                                "type": "factual"
                            })
                
                # Validate and prepare training data
                test_data = [d for d in formatted_data if d["prompt"].strip()][:100]
                
                if not test_data:
                    raise ValueError("No valid training data available")
                
                logger.info(f"Successfully formatted {len(test_data)} training samples")
                if test_data:
                    logger.debug(f"Sample prompt: {test_data[0]['prompt']}")
                
                return PromptOptimizationEnv(groq_client, reward_calculator, test_data, config)
                
            except Exception as e:
                logger.error(f"Error processing training data: {str(e)}")
                raise
        
        trainer = PPOTrainer(config, env_fn=create_env)
        trained_model = trainer.train()
        logger.info("Training completed successfully!")
        return trained_model
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise

def evaluate_models(config: Config):
    """Evaluate all available models"""
    logger.info("Starting model evaluation...")
    
    # Initialize components
    groq_client = GroqClient(config.GROQ_API_KEY)
    data_loader = DataLoader()
    reward_calculator = RewardCalculator(groq_client, config)
    
    # Load test data
    persona_data, hh_data, truth_data = data_loader.load_training_data()
    test_data = (persona_data + hh_data + truth_data)[:100]  # Use subset for evaluation
    
    # Load available models
    models = {}
    model_files = {
        'PPO_Final': f"{config.MODEL_SAVE_PATH}PPO_final.zip",
        'PPO_Best': f"{config.MODEL_SAVE_PATH}PPO_best/best_model.zip",
    }
    
    for model_name, model_path in model_files.items():
        if os.path.exists(model_path):
            try:
                from stable_baselines3 import PPO
                model = PPO.load(model_path)
                models[model_name] = model
                logger.info(f"Loaded model: {model_name}")
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {e}")
    
    if not models:
        logger.error("No trained models found for evaluation!")
        return
    
    # Run evaluation
    evaluator = ModelEvaluator(models, groq_client, reward_calculator, test_data)
    results_df = evaluator.evaluate_all_models()
    
    logger.info(f"Evaluation completed! Results saved with {len(results_df)} samples")
    return results_df

def run_server(config: Config, args):
    """Run the deployment server"""
    logger.info("Starting deployment server...")
    
    server = DeploymentServer()
    server.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )

def test_system(config: Config):
    """Run basic system tests"""
    logger.info("Running system tests...")
    
    try:
        # Test data loading
        logger.info("Testing data loader...")
        data_loader = DataLoader()
        persona_data, hh_data, truth_data = data_loader.load_training_data()
        assert len(persona_data) > 0, "PersonaChat data loading failed"
        assert len(hh_data) > 0, "HH-RLHF data loading failed"
        assert len(truth_data) > 0, "TruthfulQA data loading failed"
        logger.info("✓ Data loading test passed")
        
        # Test Groq client
        logger.info("Testing Groq client...")
        groq_client = GroqClient(config.GROQ_API_KEY)
        response = groq_client.get_response("Test prompt")
        assert isinstance(response, str), "Groq client test failed"
        logger.info("✓ Groq client test passed")
        
        # Test reward calculator
        logger.info("Testing reward calculator...")
        reward_calculator = RewardCalculator(groq_client, config)
        clarity_reward = reward_calculator.calculate_clarity_reward("test", "test prompt")
        assert isinstance(clarity_reward, (int, float)), "Clarity reward calculation failed"
        logger.info("✓ Reward calculator test passed")
        
        # Test environment
        logger.info("Testing environment...")
        env = PromptOptimizationEnv(
            groq_client, reward_calculator, 
            persona_data[:10], config
        )
        obs, info = env.reset()
        assert obs.shape == (config.EMBEDDING_DIM,), "Environment reset failed"
        logger.info("✓ Environment test passed")
        
        logger.info("All system tests passed! ✓")
        return True
        
    except Exception as e:
        logger.error(f"System test failed: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='RL Prompt Optimization System')
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup directories and check dependencies')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train RL models')
    train_parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    train_parser.add_argument('--update-freq', type=int, default=20, help='Update frequency')
    train_parser.add_argument('--eval-freq', type=int, default=100, help='Evaluation frequency')
    train_parser.add_argument('--save-freq', type=int, default=200, help='Save frequency')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained models')
    
    # Server command
    server_parser = subparsers.add_parser('serve', help='Run deployment server')
    server_parser.add_argument('--host', default='0.0.0.0', help='Server host')
    server_parser.add_argument('--port', type=int, default=5000, help='Server port')
    server_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run system tests')
    
    # All command (full pipeline)
    all_parser = subparsers.add_parser('all', help='Run complete pipeline (train + evaluate + serve)')
    all_parser.add_argument('--episodes', type=int, default=500, help='Number of training episodes')
    all_parser.add_argument('--skip-training', action='store_true', help='Skip training if models exist')
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = Config()
    
    # Print system info
    print("="*60)
    print("RL PROMPT OPTIMIZATION SYSTEM")
    print("="*60)
    print(f"Started at: {datetime.now()}")
    print(f"Command: {args.command}")
    print(f"Groq API Key: {'Set' if config.GROQ_API_KEY else 'Not Set (using mock)'}")
    print("="*60)
    
    try:
        if args.command == 'setup':
            setup_directories(config)
            logger.info("Setup completed successfully!")
            
        elif args.command == 'train':
            setup_directories(config)
            trained_model = train_models(config, args)
            
        elif args.command == 'evaluate':
            results = evaluate_models(config)
            
        elif args.command == 'serve':
            run_server(config, args)
            
        elif args.command == 'test':
            setup_directories(config)
            success = test_system(config)
            if not success:
                sys.exit(1)
                
        elif args.command == 'all':
            # Run complete pipeline
            logger.info("Running complete pipeline...")
            
            # Setup
            setup_directories(config)
            
            # Test system first
            if not test_system(config):
                logger.error("System tests failed! Aborting pipeline.")
                sys.exit(1)
            
            # Check if we should skip training
            model_exists = os.path.exists(f"{config.MODEL_SAVE_PATH}PPO_final.zip")
            if args.skip_training and model_exists:
                logger.info("Skipping training - model already exists")
            else:
                # Train models
                train_models(config, args)
            
            # Evaluate models
            evaluate_models(config)
            
            # Start server
            logger.info("Pipeline completed! Starting server...")
            run_server(config, args)
            
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()