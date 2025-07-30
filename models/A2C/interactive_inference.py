import torch
from sentence_transformers import SentenceTransformer
from utils import PromptEnvironment
from model import A2CAgent
import os

def main():
    print("🎯 A2C Interactive Inference")
    print("=" * 40)
    
    # Setup model
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    
    try:
        env = PromptEnvironment(encoder)
    except ValueError as e:
        print(f"❌ Environment setup failed: {e}")
        print("\n📝 Please ensure your .env file is properly configured.")
        return
    
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
    
    # Test API connection first
    print("🔗 Testing API connection...")
    success, message = env.test_api_connection()
    if not success:
        print(f"❌ {message}")
        print("Please check your .env file and internet connection.")
        return
    print(f"✅ {message}")
    
    print("\n🎯 Interactive Prompt Optimization")
    print("Enter prompts to optimize. Type 'quit' to exit.")
    print("-" * 40)
    
    while True:
        # Get user input
        prompt = input("\n📝 Enter a prompt to optimize: ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not prompt:
            print("Please enter a valid prompt.")
            continue
        
        try:
            # Set the original prompt
            env.original_prompt = prompt
            
            # Encode the prompt
            state = env.encode(prompt).unsqueeze(0).to(device)
            
            # Get A2C action
            action, log_prob, value = agent.select_action(state)
            action = torch.tensor(action, dtype=torch.float32).to(device)
            
            # Decode the action to get optimized prompt
            optimized_prompt = env.decode(action.squeeze())
            
            # Get LLM response
            response = env.real_llm_response(optimized_prompt)
            
            # Calculate reward
            reward = env.calculate_reward_with_feedback(prompt, optimized_prompt, response)
            
            # Display results
            print(f"\n📝 Original Prompt: {prompt}")
            print(f"🔄 Optimized Prompt: {optimized_prompt}")
            print(f"🤖 LLM Response: {response}")
            print(f"📊 Reward: {reward:.3f}")
            
            # Ask for user feedback
            feedback = input("\n❓ Are you satisfied with this optimization? (y/n): ").strip().lower()
            if feedback in ['y', 'n']:
                user_satisfied = feedback == 'y'
                # Store feedback
                env.user_feedback_history.append({
                    'original': prompt,
                    'refined': optimized_prompt,
                    'response': response,
                    'satisfied': user_satisfied,
                    'timestamp': len(env.user_feedback_history)
                })
                
                if user_satisfied:
                    print("✅ Great! Feedback recorded.")
                else:
                    print("❌ Noted. The model will learn from this feedback.")
            else:
                print("⏭️  Skipping feedback.")
                
        except Exception as e:
            print(f"❌ Error processing prompt: {str(e)}")
            continue
    
    # Show final statistics
    if env.user_feedback_history:
        stats = env.get_feedback_statistics()
        if isinstance(stats, dict):
            print(f"\n📈 Session Summary:")
            print(f"   Total prompts processed: {stats['total_feedback']}")
            print(f"   User satisfaction rate: {stats['satisfaction_rate']:.1f}%")
            print(f"   Satisfied responses: {stats['satisfied_count']}/{stats['total_feedback']}")

if __name__ == "__main__":
    main() 