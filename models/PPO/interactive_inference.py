import torch
from sentence_transformers import SentenceTransformer
from utils import PromptEnvironment
from model import PPOAgent
import os

def load_trained_model(device):
    """Load the trained PPO model"""
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    state_dim = encoder.get_sentence_embedding_dimension()
    action_dim = state_dim
    
    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, device=device)
    
    # Try to load saved model
    model_path = "saved_model/ppo_actor.pth"
    if os.path.exists(model_path):
        agent.load(model_path)
        print("✅ Loaded trained model from saved_model/ppo_actor.pth")
    else:
        print("⚠️  No trained model found. Using untrained model.")
    
    return agent, encoder

def interactive_inference():
    """Interactive inference with user feedback"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Load model and environment
    agent, encoder = load_trained_model(device)
    env = PromptEnvironment(encoder)
    
    print("\n🎯 Interactive Prompt Optimization")
    print("=" * 50)
    print("Enter your prompts and provide feedback on the optimizations.")
    print("Type 'quit' to exit.\n")
    
    while True:
        # Get user prompt
        user_prompt = input("📝 Enter your prompt (or 'quit' to exit): ").strip()
        
        if user_prompt.lower() == 'quit':
            break
        
        if not user_prompt:
            print("Please enter a valid prompt.")
            continue
        
        print(f"\n🔄 Optimizing: '{user_prompt}'")
        
        # Set the prompt and encode
        env.original_prompt = user_prompt
        state = env.encode(user_prompt).unsqueeze(0).to(device)
        
        # Get model's optimization
        action, log_prob, value = agent.select_action(state)
        action = torch.tensor(action, dtype=torch.float32).to(device)
        
        # Generate refined prompt
        refined_prompt = env.decode(action.squeeze())
        response = env.real_llm_response(refined_prompt)
        
        # Show results
        print(f"\n📝 Original: {user_prompt}")
        print(f"🔄 Optimized: {refined_prompt}")
        print(f"🤖 LLM Response: {response}")
        
        # Get user feedback
        while True:
            feedback = input("\n❓ Are you satisfied with this optimization? (y/n): ").strip().lower()
            if feedback in ['y', 'n']:
                break
            print("Please enter 'y' for yes or 'n' for no.")
        
        # Store feedback for potential future training
        env.user_feedback_history.append({
            'original': user_prompt,
            'refined': refined_prompt,
            'response': response,
            'satisfied': feedback == 'y',
            'timestamp': len(env.user_feedback_history)
        })
        
        print(f"✅ Feedback recorded: {'Satisfied' if feedback == 'y' else 'Not satisfied'}")
        
        # Show current statistics
        stats = env.get_feedback_statistics()
        if isinstance(stats, dict):
            print(f"📊 Overall satisfaction rate: {stats['satisfaction_rate']:.1f}%")
        
        print("\n" + "="*50 + "\n")
    
    # Final summary
    stats = env.get_feedback_statistics()
    if isinstance(stats, dict) and stats['total_feedback'] > 0:
        print("📈 Session Summary:")
        print(f"   Total prompts tested: {stats['total_feedback']}")
        print(f"   Satisfaction rate: {stats['satisfaction_rate']:.1f}%")
        print(f"   Satisfied: {stats['satisfied_count']}, Not satisfied: {stats['total_feedback'] - stats['satisfied_count']}")
    
    print("👋 Thank you for testing the PPO prompt optimizer!")

def main():
    print("🚀 PPO Interactive Inference")
    print("=" * 50)
    
    try:
        interactive_inference()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Exiting...")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("Please check your API keys and try again.")

if __name__ == "__main__":
    main() 