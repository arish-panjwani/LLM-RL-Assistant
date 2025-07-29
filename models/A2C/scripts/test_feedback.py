#!/usr/bin/env python3
"""
Simple terminal-based feedback system for testing A2C learning.
This bypasses the demo interface and lets you test feedback directly.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.prompt_optimizer import PromptOptimizer
from utils.config import Config
from utils.groq_client import GroqClient

def test_feedback_system():
    """Test the feedback system directly."""
    print("🧪 A2C Feedback Testing System")
    print("=" * 50)
    
    # Load configuration
    config = Config()
    model_config = config.get_model_config()
    
    # Initialize Groq client
    print("🔧 Initializing Groq API...")
    groq_api_key = os.getenv('GROQ_API_KEY')
    
    if groq_api_key:
        try:
            groq_client = GroqClient(api_key=groq_api_key)
            print("✅ Groq API initialized successfully!")
            print(f"🔑 API Key: {groq_api_key[:10]}...{groq_api_key[-4:]}")
            
            # Test connection
            if groq_client.test_connection():
                print("✅ Groq API connection test successful!")
            else:
                print("⚠️ Groq API connection test failed, but continuing...")
                
        except Exception as e:
            print(f"❌ Failed to initialize Groq API: {e}")
            groq_client = None
    else:
        print("⚠️ GROQ_API_KEY not found in environment variables")
        print("💡 To use real LLM responses, set your Groq API key:")
        print("   export GROQ_API_KEY='your-api-key-here'")
        print("   or create a .env file with: GROQ_API_KEY=your-api-key-here")
        groq_client = None
    
    # Initialize optimizer
    model_path = "data/models/a2c_domain_agnostic_best.pth"
    optimizer = PromptOptimizer(model_path, model_config, groq_client)
    
    print("✅ Optimizer initialized")
    print("✅ Model loaded")
    print()
    
    while True:
        print("\n" + "="*50)
        print("Enter your prompt (or 'quit' to exit):")
        original_prompt = input("> ").strip()
        
        if original_prompt.lower() == 'quit':
            break
            
        if not original_prompt:
            print("❌ Please enter a prompt")
            continue
        
        print(f"\n🔄 Optimizing: '{original_prompt}'")
        
        try:
            # Optimize the prompt
            result = optimizer.optimize_prompt(original_prompt, max_iterations=1)
            
            print(f"\n📝 Original: {result['original_prompt']}")
            print(f"🚀 Optimized: {result['optimized_prompt']}")
            print(f"📊 Improvement: {result.get('total_improvement', 0):.3f}")
            
            # Get LLM response for the optimized prompt
            print(f"\n🤖 Getting LLM response...")
            llm_response = optimizer._get_llm_response(result['optimized_prompt'])
            
            if groq_client:
                print(f"�� Real LLM Response:")
                print("=" * 60)
                print(llm_response)
                print("=" * 60)
            else:
                print(f"💬 Mock Response:")
                print("=" * 60)
                print(llm_response)
                print("=" * 60)
                print("⚠️ Using mock response - set GROQ_API_KEY for real responses")
            
            # Get user feedback on the RESPONSE (not the prompt)
            print(f"\n🎯 Rate the LLM RESPONSE quality:")
            print("Note: If the response seems incomplete, you can still rate it based on what you see.")
            print("1. 👍 Good response")
            print("2. 😐 Okay response") 
            print("3. 👎 Bad response")
            print("4. ⏭️ Skip feedback")
            
            while True:
                choice = input("Enter your choice (1-4): ").strip()
                
                if choice == '1':
                    feedback = 1.0
                    feedback_name = "Good response"
                    break
                elif choice == '2':
                    feedback = 0.0
                    feedback_name = "Okay response"
                    break
                elif choice == '3':
                    feedback = -1.0
                    feedback_name = "Bad response"
                    break
                elif choice == '4':
                    feedback = None
                    feedback_name = "Skipped"
                    break
                else:
                    print("❌ Please enter 1, 2, 3, or 4")
            
            if feedback is not None:
                print(f"\n🔄 Processing feedback: {feedback_name} ({feedback})")
                print(f"📝 Rating the response to: '{result['optimized_prompt']}'")
                
                # Store the original optimization result before updating
                original_result = result.copy()
                
                # Update model with feedback on the response
                feedback_result = optimizer.update_with_human_feedback(
                    original_result['original_prompt'], 
                    original_result['optimized_prompt'], 
                    feedback
                )
                
                print(f"✅ Feedback processed!")
                print(f"📊 Reward: {feedback_result.get('reward', 0):.3f}")
                print(f"🔄 Model Updated: {feedback_result.get('model_updated', False)}")
                
                if feedback_result.get('learning_stats'):
                    stats = feedback_result['learning_stats']
                    print(f"📈 Learning Stats:")
                    print(f"   Actor Loss: {stats.get('actor_loss', 0):.4f}")
                    print(f"   Critic Loss: {stats.get('critic_loss', 0):.4f}")
                    print(f"   Action Taken: {stats.get('action', 0)}")
                
                # Save feedback to history file (same as Flask app)
                try:
                    import json
                    from datetime import datetime
                    history_file = Path(__file__).parent.parent / "demo" / "optimization_history.json"
                    
                    # Load existing history
                    history = []
                    if history_file.exists():
                        with open(history_file, 'r') as f:
                            history = json.load(f)
                    
                    # Add feedback entry
                    feedback_entry = {
                        'id': len(history) + 1,
                        'timestamp': datetime.now().isoformat(),
                        'original_prompt': original_result['original_prompt'],
                        'optimized_prompt': original_result['optimized_prompt'],
                        'human_feedback': feedback,
                        'feedback_name': feedback_name,
                        'reward': feedback_result.get('reward', 0.0),
                        'model_updated': feedback_result.get('model_updated', False),
                        'learning_stats': feedback_result.get('learning_stats', {}),
                        'source': 'terminal_script'
                    }
                    
                    history.append(feedback_entry)
                    
                    # Save updated history
                    history_file.parent.mkdir(exist_ok=True)
                    with open(history_file, 'w') as f:
                        json.dump(history, f, indent=2)
                    
                    print(f"💾 Feedback saved to history file")
                    
                except Exception as e:
                    print(f"⚠️ Could not save to history: {e}")
            else:
                print("⏭️ Feedback skipped")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
        
        # Ask if user wants to continue
        print(f"\nContinue testing? (y/n): ", end="")
        if input().lower() != 'y':
            break
    
    print("\n🎉 Feedback testing completed!")
    print("💾 Model has been saved with your feedback")

if __name__ == "__main__":
    test_feedback_system() 