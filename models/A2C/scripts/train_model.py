#!/usr/bin/env python3
"""
Main A2C Training Script - Lightweight Version with Real LLM Integration
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main training function with real LLM integration."""
    parser = argparse.ArgumentParser(description="Train A2C model for prompt optimization")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to resume training from checkpoint")
    parser.add_argument("--episodes", type=int, default=50,
                       help="Number of training episodes")
    parser.add_argument("--max_steps", type=int, default=10,
                       help="Maximum steps per episode")
    parser.add_argument("--model_name", type=str, default="a2c_llm_optimized",
                       help="Name for the trained model")
    parser.add_argument("--fast_mode", action="store_true",
                       help="Fast training mode (no LLM calls during training)")
    
    args = parser.parse_args()
    
    print("🚀 A2C Training Started")
    print("=" * 40)
    
    # Import components with error handling
    try:
        from utils.config import Config
        config = Config(args.config)
        print("✅ Configuration loaded")
    except Exception as e:
        print(f"⚠️ Config loading failed: {e}")
        print("Using default configuration...")
        config = None
    
    try:
        from models.a2c_model import A2CModel
        print("✅ A2C model imported")
    except ImportError as e:
        print(f"❌ A2C model import failed: {e}")
        return
    
    try:
        from training.environment import PromptOptimizationEnv
        print("✅ Environment imported")
    except ImportError as e:
        print(f"❌ Environment import failed: {e}")
        return
    
    try:
        from utils.groq_client import GroqClient
        print("✅ Groq client imported")
    except ImportError as e:
        print(f"⚠️ Groq client import failed: {e}")
        GroqClient = None
    
    try:
        from utils.evaluation_metrics import PromptEvaluator
        print("✅ Evaluation metrics imported")
    except ImportError as e:
        print(f"❌ Evaluation metrics import failed: {e}")
        return
    
    # Initialize components
    print("Initializing components...")
    
    # Initialize Groq client
    groq_client = None
    if GroqClient:
        try:
            groq_client = GroqClient(
                api_key=config.get('api.groq_api_key') if config else None,
                base_url="https://api.groq.com/openai/v1"
            )
            print("✅ Groq client initialized")
        except Exception as e:
            print(f"⚠️ Groq client initialization failed: {e}")
            groq_client = None
    
    # Initialize evaluator with real APIs
    try:
        evaluator = PromptEvaluator(
            groq_client=groq_client,
            use_external_apis=True  # Enable real APIs for dynamic evaluation
        )
        print("✅ Evaluator initialized with real APIs")
    except Exception as e:
        print(f"❌ Evaluator initialization failed: {e}")
        return
    
    # Initialize model
    try:
        model_config = config.get_model_config() if config else {'state_dim': 30, 'action_dim': 5, 'device': 'cpu'}
        model = A2CModel(
            state_dim=model_config.get('state_dim', 30),
            action_dim=model_config.get('action_dim', 5),
            hidden_dims=model_config.get('hidden_dims', [64, 32]),
            learning_rate=model_config.get('learning_rate', 0.001),
            gamma=model_config.get('gamma', 0.99),
            device=str(model_config.get('device', 'cpu'))
        )
        print("✅ A2C model initialized")
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return
    
    # Load existing model if resuming
    if args.resume and os.path.exists(args.resume):
        try:
            model.load_model(args.resume)
            print(f"✅ Resumed training from {args.resume}")
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}")
    
    # Initialize environment with real LLM integration
    try:
        env_config = config.get_environment_config() if config else {'max_steps': 10}
        
        # Use fast mode if requested
        if args.fast_mode:
            print("⚡ Fast mode enabled - no LLM calls during training")
            groq_client = None  # Disable LLM calls for speed
        
        env = PromptOptimizationEnv(
            groq_client=groq_client,
            evaluator=evaluator,
            config=env_config
        )
        print("✅ Environment initialized with LLM integration")
    except Exception as e:
        print(f"❌ Environment initialization failed: {e}")
        return
    
    # Training parameters
    episodes = args.episodes or config.get('training.episodes', 50) if config else 50
    max_steps = args.max_steps or config.get('training.max_steps_per_episode', 10) if config else 10
    
    print(f"Training: {episodes} episodes, {max_steps} steps/episode")
    print(f"Model: {model.state_dim} → {model.action_dim} actions")
    print(f"LLM: {'Enabled' if groq_client else 'Disabled'}")
    print("=" * 40)
    
    episode_rewards = []
    episode_lengths = []
    improvement_scores = []
    
    # Progress bar
    progress_bar = tqdm(range(episodes), desc="Training Episodes")
    
    for episode in progress_bar:
        # Reset environment
        state = env.reset()
        total_reward = 0.0
        episode_length = 0
        episode_improvements = []
        
        # Episode buffer
        states, actions, rewards, next_states, dones, log_probs = [], [], [], [], [], []
        
        # Run episode
        for step in range(max_steps):
            # Select action
            action, log_prob = model.select_action(torch.FloatTensor(state))
            
            # Take step in environment (this will use real LLM evaluation)
            next_state, reward, done, info = env.step(action)
            
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
            
            total_reward += reward
            episode_length += 1
            state = next_state
            
            if done:
                break
        
        # Update model if we have enough transitions
        if len(states) > 0:
            model.update_model(states, actions, rewards, next_states, dones, log_probs)
        
        # Store episode results
        episode_rewards.append(total_reward)
        episode_lengths.append(episode_length)
        if episode_improvements:
            improvement_scores.append(np.mean(episode_improvements))
        
        # Update progress bar
        avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else total_reward
        avg_improvement = np.mean(improvement_scores[-10:]) if improvement_scores else 0.0
        
        progress_bar.set_postfix({
            'Episode': episode + 1,
            'Reward': f"{total_reward:.3f}",
            'Avg Reward': f"{avg_reward:.3f}",
            'Avg Improvement': f"{avg_improvement:.3f}",
            'Length': episode_length
        })
        
        # Save model periodically
        if (episode + 1) % 10 == 0:
            model_path = f"data/models/{args.model_name}_checkpoint_episode_{episode + 1}.pth"
            model.save_model(model_path)
            print(f"\n💾 Saved checkpoint at episode {episode + 1}")
    
    # Save final model
    final_model_path = f"data/models/{args.model_name}_final.pth"
    best_model_path = f"data/models/{args.model_name}_best.pth"
    
    model.save_model(final_model_path)
    model.save_model(best_model_path)
    
    print("\n" + "=" * 60)
    print("🎉 Training Completed!")
    print(f"📊 Final Statistics:")
    print(f"   Total episodes: {len(episode_rewards)}")
    print(f"   Average reward: {np.mean(episode_rewards):.3f}")
    print(f"   Best reward: {np.max(episode_rewards):.3f}")
    print(f"   Average episode length: {np.mean(episode_lengths):.1f}")
    if improvement_scores:
        print(f"   Average improvement: {np.mean(improvement_scores):.3f}")
        print(f"   Best improvement: {np.max(improvement_scores):.3f}")
    print(f"   Final model saved to: {final_model_path}")
    print(f"   Best model saved to: {best_model_path}")
    
    print("\n✅ You can now use the trained model!")
    print(f"   Run: python scripts/evaluate_model.py --model_path {best_model_path}")
    print(f"   Run: python scripts/optimize_prompts.py --model_path {best_model_path}")

if __name__ == "__main__":
    main() 