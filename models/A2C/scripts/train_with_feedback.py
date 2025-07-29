#!/usr/bin/env python3
"""
Train A2C model with collected human feedback data.
"""

import json
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.prompt_optimizer import PromptOptimizer
from utils.config import Config

def load_feedback_data(feedback_file: str = "demo/optimization_history.json"):
    """Load collected human feedback data."""
    try:
        with open(feedback_file, 'r') as f:
            data = json.load(f)
        
        # Filter for entries with human feedback
        feedback_data = []
        for entry in data:
            if 'human_feedback' in entry and entry['human_feedback'] is not None:
                feedback_data.append(entry)
        
        print(f"Loaded {len(feedback_data)} feedback entries")
        return feedback_data
    except Exception as e:
        print(f"Error loading feedback data: {e}")
        return []

def train_with_feedback(optimizer: PromptOptimizer, feedback_data: list, epochs: int = 10):
    """Train the model with collected feedback data."""
    print(f"Training with {len(feedback_data)} feedback samples for {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_losses = []
        
        for entry in feedback_data:
            try:
                original_prompt = entry['original_prompt']
                optimized_prompt = entry['optimized_prompt']
                human_feedback = entry['human_feedback']
                
                # Update model with this feedback
                result = optimizer.update_with_human_feedback(
                    original_prompt, optimized_prompt, human_feedback
                )
                
                if result.get('learning_stats'):
                    epoch_losses.append(result['learning_stats'].get('actor_loss', 0))
                
            except Exception as e:
                print(f"Error processing feedback entry: {e}")
                continue
        
        # Calculate average loss for this epoch
        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch + 1}/{epochs} - Average Actor Loss: {avg_loss:.4f}")
        else:
            print(f"Epoch {epoch + 1}/{epochs} - No valid feedback processed")
    
    print("Training completed!")

def evaluate_model_performance(optimizer: PromptOptimizer, test_prompts: list):
    """Evaluate model performance on test prompts."""
    print("Evaluating model performance...")
    
    results = []
    for prompt in test_prompts:
        try:
            result = optimizer.optimize_prompt(prompt, max_iterations=1)
            results.append({
                'prompt': prompt,
                'optimized_prompt': result['optimized_prompt'],
                'improvement': result['total_improvement']
            })
        except Exception as e:
            print(f"Error evaluating prompt '{prompt}': {e}")
    
    # Calculate average improvement
    improvements = [r['improvement'] for r in results if r['improvement'] is not None]
    if improvements:
        avg_improvement = np.mean(improvements)
        print(f"Average improvement: {avg_improvement:.3f}")
        print(f"Best improvement: {max(improvements):.3f}")
        print(f"Worst improvement: {min(improvements):.3f}")
    
    return results

def main():
    """Main training function."""
    print("🚀 Starting A2C Model Training with Human Feedback")
    
    # Load configuration
    config = Config()
    model_config = config.get_model_config()
    
    # Initialize optimizer
    model_path = "data/models/a2c_domain_agnostic_best.pth"
    optimizer = PromptOptimizer(model_path, model_config)
    
    # Load feedback data
    feedback_data = load_feedback_data()
    
    if not feedback_data:
        print("❌ No feedback data found. Please collect some feedback first.")
        return
    
    # Train with feedback
    train_with_feedback(optimizer, feedback_data, epochs=5)
    
    # Evaluate performance
    test_prompts = [
        "What is machine learning?",
        "Explain neural networks",
        "How does AI work?",
        "What is deep learning?",
        "Explain reinforcement learning"
    ]
    
    evaluation_results = evaluate_model_performance(optimizer, test_prompts)
    
    # Save evaluation results
    with open("data/feedback_training_results.json", 'w') as f:
        json.dump({
            'feedback_samples': len(feedback_data),
            'test_results': evaluation_results,
            'training_completed': True
        }, f, indent=2)
    
    print("✅ Training and evaluation completed!")
    print("📊 Results saved to: data/feedback_training_results.json")

if __name__ == "__main__":
    main() 