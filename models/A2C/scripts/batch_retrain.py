#!/usr/bin/env python3
"""
Full batch retraining with collected feedback.
Run this every 50 feedback samples or weekly.
"""

import json
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.prompt_optimizer import PromptOptimizer
from utils.config import Config

def load_feedback_data():
    """Load all collected feedback data."""
    try:
        with open("demo/optimization_history.json", 'r') as f:
            data = json.load(f)
        
        # Filter for entries with human feedback
        feedback_data = []
        for entry in data:
            if 'human_feedback' in entry and entry['human_feedback'] is not None:
                feedback_data.append(entry)
        
        return feedback_data
    except Exception as e:
        print(f"Error loading feedback data: {e}")
        return []

def batch_retrain(epochs: int = 10):
    """Perform full batch retraining."""
    print("🚀 Starting Full Batch Retraining")
    
    # Load configuration
    config = Config()
    model_config = config.get_model_config()
    
    # Initialize optimizer with proper model path handling
    model_path = "data/models/a2c_domain_agnostic_best.pth"
    
    # Check if model file exists
    if Path(model_path).exists():
        print(f"📁 Loading existing model from: {model_path}")
        optimizer = PromptOptimizer(model_path, model_config)
    else:
        print(f"📁 No existing model found. Creating new model...")
        # Create models directory if it doesn't exist
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        # Initialize with None to create a new model
        optimizer = PromptOptimizer(None, model_config)
    
    # Load feedback data
    feedback_data = load_feedback_data()
    
    if len(feedback_data) < 10:
        print(f"❌ Not enough feedback data ({len(feedback_data)} samples). Need at least 10.")
        return
    
    print(f"📊 Training with {len(feedback_data)} feedback samples")
    
    # Training loop
    for epoch in range(epochs):
        epoch_losses = []
        epoch_rewards = []
        
        print(f"\n📚 Epoch {epoch + 1}/{epochs}")
        
        for i, entry in enumerate(feedback_data):
            try:
                original_prompt = entry['original_prompt']
                optimized_prompt = entry['optimized_prompt']
                human_feedback = entry['human_feedback']
                
                # Update model with this feedback
                result = optimizer.update_with_human_feedback(
                    original_prompt, optimized_prompt, human_feedback
                )
                
                if result.get('learning_stats'):
                    learning_stats = result['learning_stats']
                    epoch_losses.append(learning_stats.get('actor_loss', 0))
                    epoch_rewards.append(result.get('reward', 0))
                
                # Progress indicator
                if (i + 1) % 10 == 0:
                    print(f"   Processed {i + 1}/{len(feedback_data)} samples")
                
            except Exception as e:
                print(f"   Error processing sample {i}: {e}")
                continue
        
        # Epoch summary
        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            avg_reward = np.mean(epoch_rewards)
            print(f"   📈 Avg Actor Loss: {avg_loss:.4f}")
            print(f"   🎯 Avg Reward: {avg_reward:.3f}")
        else:
            print("   ⚠️ No valid samples processed this epoch")
    
    # Save the retrained model
    if optimizer.model:
        # Ensure the directory exists
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        optimizer.model.save_model(model_path)
        print(f"\n💾 Model saved to: {model_path}")
    else:
        print("\n❌ No model to save - training failed")
        return
    
    # Performance evaluation
    print("\n🔍 Evaluating Retrained Model...")
    evaluate_retrained_model(optimizer)

def evaluate_retrained_model(optimizer):
    """Evaluate the retrained model."""
    test_prompts = [
        "What is artificial intelligence?",
        "Explain deep learning concepts",
        "How do neural networks work?",
        "What is machine learning?",
        "Explain reinforcement learning"
    ]
    
    improvements = []
    
    for prompt in test_prompts:
        try:
            result = optimizer.optimize_prompt(prompt, max_iterations=1)
            improvement = result.get('total_improvement', 0)
            improvements.append(improvement)
            
            print(f"   '{prompt[:30]}...' → Improvement: {improvement:.3f}")
            
        except Exception as e:
            print(f"   Error testing '{prompt}': {e}")
    
    if improvements:
        avg_improvement = np.mean(improvements)
        print(f"\n📊 Average Improvement: {avg_improvement:.3f}")
        
        # Performance rating
        if avg_improvement > 0.3:
            print("🎉 Excellent! Model has improved significantly!")
        elif avg_improvement > 0.1:
            print("👍 Good improvement!")
        elif avg_improvement > 0:
            print("😐 Slight improvement")
        else:
            print("❌ No improvement detected")

def main():
    """Main function."""
    print("🔄 Full Batch Retraining with Human Feedback")
    print("=" * 50)
    
    # Check if we have enough data
    feedback_data = load_feedback_data()
    print(f"📊 Found {len(feedback_data)} feedback samples")
    
    if len(feedback_data) < 10:
        print("❌ Need at least 10 feedback samples for batch training.")
        print("💡 Continue using the demo to collect more feedback!")
        return
    
    # Perform batch training
    batch_retrain(epochs=5)
    
    print("\n✅ Batch retraining completed!")
    print("🎯 Your model should now be better at optimizing prompts!")

if __name__ == "__main__":
    main() 