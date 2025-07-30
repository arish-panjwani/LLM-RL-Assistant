#!/usr/bin/env python3
"""
A2C Model Deployment Script
"""

import os
import sys
import torch
from sentence_transformers import SentenceTransformer
from utils import PromptEnvironment
from model import A2CAgent

def main():
    print("🚀 A2C Model Deployment")
    print("=" * 40)
    
    # Setup model
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    env = PromptEnvironment(encoder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    state_dim = encoder.get_sentence_embedding_dimension()
    action_dim = state_dim
    
    agent = A2CAgent(state_dim=state_dim, action_dim=action_dim, device=device)
    
    # Load trained model
    model_path = "saved_model/a2c_actor.pth"
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"✅ Loaded trained A2C model from {model_path}")
    else:
        print("⚠️  No trained model found. Using untrained model.")
        print("   Train the model first using main.py")
    
    print("\n🎯 A2C Model Ready for Deployment")
    print("Model can now be used for real-time prompt optimization.")
    print("-" * 40)
    
    # Test the model
    test_prompt = "Hello world"
    print(f"\n🧪 Testing with prompt: '{test_prompt}'")
    
    try:
        env.original_prompt = test_prompt
        state = env.encode(test_prompt).unsqueeze(0).to(device)
        action, log_prob, value = agent.select_action(state)
        action = torch.tensor(action, dtype=torch.float32).to(device)
        optimized_prompt = env.decode(action.squeeze())
        
        print(f"✅ Test successful!")
        print(f"   Original: {test_prompt}")
        print(f"   Optimized: {optimized_prompt}")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return
    
    print("\n🎯 Deployment Options:")
    print("1. Run interactive inference: python interactive_inference.py")
    print("2. Start web demo: cd demo && python run_demo.py")
    print("3. Use in your own code: from model import A2CAgent")
    
    print("\n✅ A2C model is ready for deployment!")

if __name__ == "__main__":
    main() 