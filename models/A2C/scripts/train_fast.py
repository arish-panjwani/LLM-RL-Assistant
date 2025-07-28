#!/usr/bin/env python3
"""
Super Fast A2C Training Script - No LLM Calls for Quick Testing
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
    """Super fast training function - no LLM calls."""
    parser = argparse.ArgumentParser(description="Fast A2C training for prompt optimization")
    parser.add_argument("--episodes", type=int, default=20,
                       help="Number of training episodes")
    parser.add_argument("--max_steps", type=int, default=5,
                       help="Maximum steps per episode")
    parser.add_argument("--model_name", type=str, default="a2c_fast",
                       help="Name for the trained model")
    
    args = parser.parse_args()
    
    print("⚡ Super Fast A2C Training - No LLM Calls")
    print("=" * 50)
    
    # Import components
    try:
        from models.a2c_model import A2CModel
        from training.environment import PromptOptimizationEnv
        from utils.evaluation_metrics import PromptEvaluator
        print("✅ All components imported")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return
    
    # Initialize components (no LLM)
    print("\n📦 Initializing components...")
    
    evaluator = PromptEvaluator(groq_client=None, use_external_apis=False)
    print("✅ Evaluator initialized (no APIs)")
    
    model = A2CModel(
        state_dim=30,
        action_dim=5,
        hidden_dims=[64, 32],
        learning_rate=0.001,
        gamma=0.99,
        device='cpu'
    )
    print("✅ A2C model initialized")
    
    env = PromptOptimizationEnv(
        groq_client=None,  # No LLM calls
        evaluator=evaluator,
        config={'max_steps': args.max_steps}
    )
    print("✅ Environment initialized (no LLM)")
    
    # Training parameters
    episodes = args.episodes
    max_steps = args.max_steps
    
    print(f"\n🎯 Training Parameters:")
    print(f"   Episodes: {episodes}")
    print(f"   Max steps per episode: {max_steps}")
    print(f"   Expected time: ~{episodes * max_steps * 0.1:.1f} seconds")
    
    # Training loop
    print(f"\n🏃‍♂️ Starting FAST training...")
    print("=" * 50)
    
    episode_rewards = []
    episode_lengths = []
    
    # Progress bar
    progress_bar = tqdm(range(episodes), desc="Training Episodes")
    
    for episode in progress_bar:
        # Reset environment
        state = env.reset()
        total_reward = 0.0
        episode_length = 0
        
        # Episode buffer
        states, actions, rewards, next_states, dones, log_probs = [], [], [], [], [], []
        
        # Run episode
        for step in range(max_steps):
            # Select action
            action, log_prob = model.select_action(torch.FloatTensor(state))
            
            # Take step in environment (no LLM calls)
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            log_probs.append(log_prob)
            
            total_reward += reward
            episode_length += 1
            state = next_state
            
            if done:
                break
        
        # Update model
        if len(states) > 0:
            model.update_model(states, actions, rewards, next_states, dones, log_probs)
        
        # Store episode results
        episode_rewards.append(total_reward)
        episode_lengths.append(episode_length)
        
        # Update progress bar
        avg_reward = np.mean(episode_rewards[-5:]) if len(episode_rewards) >= 5 else total_reward
        
        progress_bar.set_postfix({
            'Episode': episode + 1,
            'Reward': f"{total_reward:.3f}",
            'Avg Reward': f"{avg_reward:.3f}",
            'Length': episode_length
        })
    
    # Save final model
    final_model_path = f"data/models/{args.model_name}_final.pth"
    best_model_path = f"data/models/{args.model_name}_best.pth"
    
    model.save_model(final_model_path)
    model.save_model(best_model_path)
    
    print("\n" + "=" * 50)
    print("🎉 Fast Training Completed!")
    print(f"📊 Final Statistics:")
    print(f"   Total episodes: {len(episode_rewards)}")
    print(f"   Average reward: {np.mean(episode_rewards):.3f}")
    print(f"   Best reward: {np.max(episode_rewards):.3f}")
    print(f"   Average episode length: {np.mean(episode_lengths):.1f}")
    print(f"   Final model saved to: {final_model_path}")
    print(f"   Best model saved to: {best_model_path}")
    
    print("\n✅ You can now use the trained model!")
    print(f"   Run: python scripts/evaluate_model.py --model_path {best_model_path}")
    print(f"   Run: python scripts/optimize_prompts.py --model_path {best_model_path}")

if __name__ == "__main__":
    main() 