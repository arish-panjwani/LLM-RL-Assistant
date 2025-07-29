#!/usr/bin/env python3
"""
Quick performance check after collecting feedback.
Run this every 10 feedback samples.
"""

import json
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.prompt_optimizer import PromptOptimizer
from utils.config import Config
from utils.groq_client import GroqClient

def quick_performance_check():
    """Quick check of model performance on user's actual feedback data."""
    print("🔍 Quick Performance Check - Testing Your Learning")
    
    # Load configuration
    config = Config()
    model_config = config.get_model_config()
    
    # Initialize Groq client
    groq_api_key = os.getenv('GROQ_API_KEY')
    groq_client = None
    if groq_api_key:
        try:
            groq_client = GroqClient(api_key=groq_api_key)
            print("✅ Groq API available for testing")
        except:
            print("⚠️ Groq API not available, using mock responses")
    
    # Initialize optimizer
    model_path = "data/models/a2c_domain_agnostic_best.pth"
    optimizer = PromptOptimizer(model_path, model_config, groq_client)
    
    # Load user's feedback history
    history_file = Path(__file__).parent.parent / "demo" / "optimization_history.json"
    user_prompts = []
    feedback_data = []
    
    if history_file.exists():
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
                # Get unique prompts from user's feedback history
                user_prompts = list(set([item.get('original_prompt', '') for item in history if item.get('original_prompt')]))
                feedback_data = [item for item in history if item.get('human_feedback') is not None]
                print(f"📚 Found {len(user_prompts)} unique prompts from your feedback history")
                print(f"📊 Found {len(feedback_data)} feedback samples (Flask + Terminal)")
                
                # Show feedback sources
                flask_feedback = [item for item in feedback_data if item.get('source') != 'terminal_script']
                terminal_feedback = [item for item in feedback_data if item.get('source') == 'terminal_script']
                print(f"   - Flask app feedback: {len(flask_feedback)}")
                print(f"   - Terminal script feedback: {len(terminal_feedback)}")
                
        except Exception as e:
            print(f"⚠️ Could not load feedback history: {e}")
    
    # If no user prompts, use some default ones but inform user
    if not user_prompts:
        print("⚠️ No feedback history found, using default test prompts")
        test_prompts = [
            "What is AI?",
            "Explain machine learning", 
            "How does neural networks work?"
        ]
    else:
        # Use user's actual prompts (up to 5 for quick testing)
        test_prompts = user_prompts[:5]
        print(f"🎯 Testing on YOUR actual prompts from feedback history")
    
    print("\n📝 Testing Model Performance on Your Data:")
    print("-" * 60)
    
    total_improvement = 0
    successful_optimizations = 0
    response_ratings = []
    
    for i, prompt in enumerate(test_prompts, 1):
        try:
            print(f"\n{i}. Testing: '{prompt[:60]}...'")
            
            # Optimize the prompt
            result = optimizer.optimize_prompt(prompt, max_iterations=1)
            improvement = result.get('total_improvement', 0)
            
            print(f"   Original: {result['original_prompt'][:50]}...")
            print(f"   Optimized: {result['optimized_prompt'][:50]}...")
            print(f"   Improvement: {improvement:.3f}")
            
            # Get LLM response for evaluation
            llm_response = optimizer._get_llm_response(result['optimized_prompt'])
            print(f"   Response: {llm_response[:100]}...")
            
            # Simulate response quality rating (you can replace this with actual rating)
            response_quality = len(llm_response) / 200.0  # Simple length-based quality
            response_ratings.append(response_quality)
            print(f"   Response Quality: {response_quality:.3f}")
            
            total_improvement += improvement
            successful_optimizations += 1
            
        except Exception as e:
            print(f"   Error: {e}")
    
    if successful_optimizations > 0:
        avg_improvement = total_improvement / successful_optimizations
        avg_response_quality = sum(response_ratings) / len(response_ratings)
        
        print(f"\n📊 Performance Summary:")
        print(f"   Average Improvement: {avg_improvement:.3f}")
        print(f"   Average Response Quality: {avg_response_quality:.3f}")
        print(f"   Prompts Tested: {successful_optimizations}")
        
        # Performance rating based on YOUR data
        if avg_improvement > 0.3 and avg_response_quality > 0.7:
            print("✅ Excellent performance on your data!")
        elif avg_improvement > 0.1 and avg_response_quality > 0.5:
            print("👍 Good performance on your data")
        elif avg_improvement > 0:
            print("😐 Acceptable performance on your data")
        else:
            print("❌ Needs improvement on your data")
    
    # Check training stats
    if optimizer.model:
        stats = optimizer.model.get_training_stats()
        print(f"\n📈 Your Learning Stats:")
        print(f"   Total Episodes: {stats.get('total_episodes', 0)}")
        print(f"   Avg Actor Loss: {stats.get('avg_actor_loss', 0):.4f}")
        print(f"   Avg Reward: {stats.get('avg_reward', 0):.3f}")
        
        if stats.get('total_episodes', 0) > 0:
            print(f"🎯 Your model has learned from {stats.get('total_episodes', 0)} feedback samples!")
        else:
            print("⚠️ No learning episodes found - give more feedback!")

if __name__ == "__main__":
    quick_performance_check() 