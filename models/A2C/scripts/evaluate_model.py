#!/usr/bin/env python3
"""
Evaluate the trained A2C model performance.
"""

import os
import sys
import argparse
import json
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate A2C model performance")
    parser.add_argument("--model_path", type=str, default="data/models/a2c_domain_agnostic_best.pth",
                       help="Path to the trained model")
    parser.add_argument("--config_path", type=str, default="config/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--output_path", type=str, default="data/evaluation_results.json",
                       help="Path to save evaluation results")
    parser.add_argument("--test_prompts", type=int, default=10,
                       help="Number of test prompts to evaluate")
    
    args = parser.parse_args()
    
    print("🔍 A2C Model Evaluation")
    print("=" * 50)
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model not found: {args.model_path}")
        print("Please train the model first using: python train_simple.py")
        return
    
    # Import components with error handling
    try:
        from utils.config import Config
        config = Config(args.config_path)
        print("✅ Configuration loaded")
    except Exception as e:
        print(f"⚠️ Config loading failed: {e}")
        print("Using default configuration...")
        config = None
    
    try:
        from models.a2c_model import A2CModel
        print("✅ A2C model imported")
    except ImportError:
        print("⚠️ A2C model import failed, using minimal version...")
        # Create minimal A2C model class
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim
        
        class MinimalA2CModel:
            def __init__(self, state_dim=30, action_dim=5, device='cpu'):
                self.state_dim = state_dim
                self.action_dim = action_dim
                self.device = device
                
                self.actor = nn.Sequential(
                    nn.Linear(state_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, action_dim)
                ).to(device)
                
                self.critic = nn.Sequential(
                    nn.Linear(state_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1)
                ).to(device)
            
            def load_model(self, filepath):
                if os.path.exists(filepath):
                    checkpoint = torch.load(filepath, map_location=self.device)
                    self.actor.load_state_dict(checkpoint['actor_state_dict'])
                    self.critic.load_state_dict(checkpoint['critic_state_dict'])
                    print(f"✅ Model loaded from {filepath}")
                else:
                    print(f"⚠️ Model file not found: {filepath}")
            
            def select_action(self, state, training=False):
                state = state.to(self.device)
                action_probs = F.softmax(self.actor(state), dim=-1)
                action = torch.argmax(action_probs)
                return action.item(), torch.log(action_probs[action])
        
        A2CModel = MinimalA2CModel
    
    try:
        from utils.evaluation_metrics import PromptEvaluator
        print("✅ Evaluation metrics imported")
    except ImportError as e:
        print(f"⚠️ Evaluation metrics import failed: {e}")
        print("Using minimal evaluation...")
        
        class MinimalPromptEvaluator:
            def __init__(self, use_external_apis=False):
                self.use_external_apis = use_external_apis
            
            def extract_prompt_features(self, prompt):
                import torch
                features = []
                features.append(len(prompt) / 200.0)
                features.append(len(prompt.split()) / 50.0)
                features.append(prompt.count('?') / 10.0)
                features.append(prompt.count('!') / 10.0)
                features.append(prompt.count('.') / 10.0)
                
                while len(features) < 30:
                    features.append(0.0)
                
                return torch.FloatTensor(features[:30])
            
            def evaluate_prompt(self, prompt):
                score = 0.0
                if 10 <= len(prompt) <= 200:
                    score += 0.3
                if '?' in prompt:
                    score += 0.2
                if 3 <= len(prompt.split()) <= 20:
                    score += 0.3
                return min(score, 1.0)
        
        PromptEvaluator = MinimalPromptEvaluator
    
    # Initialize components
    print("\n📦 Initializing components...")
    
    # Initialize model
    model_config = config.get_model_config() if config else {'state_dim': 30, 'action_dim': 5, 'device': 'cpu'}
    model = A2CModel(
        state_dim=model_config.get('state_dim', 30),
        action_dim=model_config.get('action_dim', 5),
        device=str(model_config.get('device', 'cpu'))
    )
    model.load_model(args.model_path)
    
    # Initialize evaluator
    evaluator = PromptEvaluator(use_external_apis=False)
    
    # Test prompts
    test_prompts = [
        "What is machine learning?",
        "How does neural network work?",
        "Explain reinforcement learning",
        "What is deep learning?",
        "How to train a model?",
        "What is AI?",
        "Explain computer vision",
        "What is natural language processing?",
        "How does gradient descent work?",
        "What is overfitting?"
    ][:args.test_prompts]
    
    print(f"\n🧪 Evaluating {len(test_prompts)} test prompts...")
    print("=" * 50)
    
    results = []
    total_score = 0.0
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n📝 Test {i+1}/{len(test_prompts)}: {prompt[:50]}...")
        
        # Extract features
        features = evaluator.extract_prompt_features(prompt)
        
        # Get model action
        action, _ = model.select_action(features, training=False)
        
        # Apply action to prompt
        modified_prompt = apply_action_to_prompt(prompt, action)
        
        # Evaluate original and modified prompts
        original_score = evaluator.evaluate_prompt(prompt)
        modified_score = evaluator.evaluate_prompt(modified_prompt)
        
        improvement = modified_score - original_score
        
        result = {
            'prompt_id': i + 1,
            'original_prompt': prompt,
            'modified_prompt': modified_prompt,
            'action_taken': action,
            'original_score': original_score,
            'modified_score': modified_score,
            'improvement': improvement
        }
        
        results.append(result)
        total_score += modified_score
        
        print(f"   Action: {action}")
        print(f"   Original Score: {original_score:.3f}")
        print(f"   Modified Score: {modified_score:.3f}")
        print(f"   Improvement: {improvement:+.3f}")
    
    # Calculate overall statistics
    avg_score = total_score / len(test_prompts)
    improvements = [r['improvement'] for r in results]
    avg_improvement = sum(improvements) / len(improvements)
    positive_improvements = len([i for i in improvements if i > 0])
    
    print("\n" + "=" * 50)
    print("📊 Evaluation Results")
    print("=" * 50)
    print(f"Total prompts evaluated: {len(test_prompts)}")
    print(f"Average score: {avg_score:.3f}")
    print(f"Average improvement: {avg_improvement:+.3f}")
    print(f"Prompts improved: {positive_improvements}/{len(test_prompts)} ({positive_improvements/len(test_prompts)*100:.1f}%)")
    
    # Save results
    evaluation_summary = {
        'model_path': args.model_path,
        'test_prompts_count': len(test_prompts),
        'average_score': avg_score,
        'average_improvement': avg_improvement,
        'prompts_improved': positive_improvements,
        'improvement_rate': positive_improvements / len(test_prompts),
        'detailed_results': results
    }
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    with open(args.output_path, 'w') as f:
        json.dump(evaluation_summary, f, indent=2)
    
    print(f"\n💾 Results saved to: {args.output_path}")
    print("✅ Evaluation completed!")

def apply_action_to_prompt(prompt, action):
    """Apply action to modify prompt - MUST MATCH TRAINING ENVIRONMENT."""
    actions = [
        lambda p: f"Please provide a clear and detailed explanation of: {p}",  # add_clarity
        lambda p: f"Please provide specific technical details about: {p}",     # add_specificity
        lambda p: f"In the context of artificial intelligence and machine learning, please explain: {p}",  # add_context
        lambda p: f"Please explain in simple terms that a beginner can understand: {p}",  # simplify_language
        lambda p: p  # no_change
    ]
    return actions[action](prompt)

if __name__ == "__main__":
    main() 