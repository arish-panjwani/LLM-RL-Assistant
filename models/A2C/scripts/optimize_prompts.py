#!/usr/bin/env python3
"""
Optimize prompts using the trained A2C model with real LLM integration.
"""

import os
import sys
import argparse
import json
from pathlib import Path
import logging
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main optimization function with real LLM integration."""
    parser = argparse.ArgumentParser(description="Optimize prompts using A2C model with LLM")
    parser.add_argument("--model_path", type=str, default="data/models/a2c_domain_agnostic_best.pth",
                        help="Path to the trained model")
    parser.add_argument("--config_path", type=str, default="config/config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--prompts", nargs='+', default=[
        # 4 DIVERSE, DOMAIN-AGNOSTIC TEST CASES (No Rate Limit Issues)
        # Business/Finance
        "How to create a business plan?",
        
        # Health/Medical
        "What are the benefits of exercise?",
        
        # Lifestyle/Fashion
        "What are the latest fashion trends?",
        
        # Academic/Research
        "How to write a research paper?"
    ], help="Prompts to optimize")
    
    args = parser.parse_args()
    
    print("🚀 A2C Prompt Optimization")
    print("=" * 40)
    
    # Load configuration
    try:
        from utils.config import Config
        config = Config(args.config_path)
        print("✅ Configuration loaded")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return
    
    # Import prompt optimizer
    try:
        from models.prompt_optimizer import PromptOptimizer
        print("✅ Prompt optimizer imported")
    except Exception as e:
        print(f"❌ Failed to import prompt optimizer: {e}")
        return
    
    # Initialize Groq client
    print("Initializing components...")
    try:
        from utils.groq_client import GroqClient
        api_config = config.get_api_config()
        groq_api_key = api_config.get('groq_api_key')
        groq_client = GroqClient(api_key=groq_api_key)
        print("✅ Groq client initialized")
    except Exception as e:
        print(f"⚠️ Groq client not available: {e}")
        groq_client = None
    
    # Initialize prompt optimizer
    try:
        optimizer = PromptOptimizer(args.model_path, config.get_model_config(), groq_client)
        print("✅ Prompt optimizer initialized")
    except Exception as e:
        print(f"❌ Failed to initialize optimizer: {e}")
        return
    
    print(f"\n🎯 Optimizing {len(args.prompts)} diverse prompts with REAL API evaluation...")
    print("=" * 60)
    
    # Optimize prompts
    results = []
    total_improvement = 0.0
    prompts_improved = 0
    
    for i, prompt in enumerate(args.prompts):
        print(f"\n📝 Optimizing prompt {i+1}/{len(args.prompts)}:")
        print(f"   Original: {prompt}")
        
        try:
            result = optimizer.optimize_prompt(prompt)
            
            print(f"   Optimized: {result['optimized_prompt']}")
            print(f"   Initial Score: {result['initial_score']:.3f}")
            print(f"   Final Score: {result['final_score']:.3f}")
            print(f"   Improvement: {result['total_improvement']:+.3f}")
            
            # Show learning summary
            if result.get('learning_summary'):
                print(f"   Learning: {list(result['learning_summary'].keys())}")
            
            results.append(result)
            total_improvement += result['total_improvement']
            
            if result['total_improvement'] > 0:
                prompts_improved += 1
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                'original_prompt': prompt,
                'optimized_prompt': prompt,
                'initial_score': 0.0,
                'final_score': 0.0,
                'total_improvement': 0.0,
                'error': str(e)
            })
    
    # Calculate statistics
    avg_improvement = total_improvement / len(args.prompts) if args.prompts else 0.0
    improvement_rate = (prompts_improved / len(args.prompts)) * 100 if args.prompts else 0.0
    
    print("\n" + "=" * 60)
    print("📊 Optimization Results Summary")
    print("=" * 60)
    print(f"Total prompts optimized: {len(args.prompts)}")
    print(f"Successful optimizations: {len([r for r in results if 'error' not in r])}")
    print(f"Average improvement: {avg_improvement:+.3f}")
    print(f"Prompts improved: {prompts_improved}/{len(args.prompts)} ({improvement_rate:.1f}%)")
    
    if results:
        improvements = [r['total_improvement'] for r in results if 'error' not in r]
        if improvements:
            print(f"Best improvement: {max(improvements):+.3f}")
            print(f"Worst improvement: {min(improvements):+.3f}")
    
    # Save results
    try:
        import json
        from pathlib import Path
        
        output_file = "data/optimized_prompts_llm.json"
        Path("data").mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
    except Exception as e:
        print(f"⚠️ Failed to save results: {e}")
    
    print("✅ Optimization completed!")

def optimize_single_prompt(prompt, optimizer, max_iterations, show_responses):
    """Optimize a single prompt using the optimizer."""
    return optimizer.optimize_prompt(prompt, max_iterations)

def get_interactive_prompts():
    """Get prompts interactively from user."""
    prompts = []
    print("\n📝 Enter prompts to optimize (press Enter twice to finish):")
    
    while True:
        prompt = input("Enter prompt: ").strip()
        if not prompt:
            if prompts:
                break
            else:
                print("Please enter at least one prompt.")
                continue
        prompts.append(prompt)
    
    return prompts

if __name__ == "__main__":
    main() 