#!/usr/bin/env python3
"""
Deployment script for PPO model - Production ready
"""

import torch
from sentence_transformers import SentenceTransformer
from utils import PromptEnvironment
from model import PPOAgent
import os
import sys

def load_model():
    """Load the trained PPO model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    state_dim = encoder.get_sentence_embedding_dimension()
    action_dim = state_dim
    
    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, device=device)
    
    # Load trained model
    model_path = "saved_model/ppo_actor.pth"
    if os.path.exists(model_path):
        agent.load(model_path)
        print("✅ Loaded trained model successfully")
        return agent, encoder, device
    else:
        print("❌ No trained model found at saved_model/ppo_actor.pth")
        print("Please train the model first using: python main.py")
        return None, None, None

def optimize_prompt(prompt, agent, encoder, device):
    """Optimize a single prompt"""
    env = PromptEnvironment(encoder)
    env.original_prompt = prompt
    
    # Encode and optimize
    state = env.encode(prompt).unsqueeze(0).to(device)
    action, _, _ = agent.select_action(state)
    action = torch.tensor(action, dtype=torch.float32).to(device)
    
    # Generate optimized prompt
    optimized_prompt = env.decode(action.squeeze())
    
    return optimized_prompt

def main():
    print("🚀 PPO Model Deployment")
    print("=" * 40)
    
    # Load model
    agent, encoder, device = load_model()
    if agent is None:
        return
    
    print("\n🎯 Model ready for deployment!")
    print("Enter prompts to optimize, or 'quit' to exit.\n")
    
    while True:
        try:
            # Get user input
            user_prompt = input("📝 Enter prompt to optimize: ").strip()
            
            if user_prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_prompt:
                print("Please enter a valid prompt.")
                continue
            
            # Optimize the prompt
            print(f"\n🔄 Optimizing: '{user_prompt}'")
            optimized = optimize_prompt(user_prompt, agent, encoder, device)
            
            print(f"📝 Original: {user_prompt}")
            print(f"🔄 Optimized: {optimized}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print("Please try again.")

if __name__ == "__main__":
    main() 