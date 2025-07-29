import torch
from sentence_transformers import SentenceTransformer
from utils import PromptEnvironment
from model import PPOAgent
import os
import sys

def main():
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    env = PromptEnvironment(encoder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim = encoder.get_sentence_embedding_dimension()
    action_dim = state_dim

    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, device=device)

    print("🚀 PPO Training with User Feedback")
    print("=" * 50)
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        mode_arg = sys.argv[1].lower()
        if mode_arg in ['train', 'training', '1']:
            mode = "1"
        elif mode_arg in ['interactive', 'feedback', '2']:
            mode = "2"
        elif mode_arg in ['deploy', 'deployment', '0']:
            mode = "deploy"
        else:
            print(f"❌ Unknown mode: {mode_arg}")
            print("Available modes: train, interactive, deploy")
            return
    else:
        # Interactive mode selection
        print("Select mode:")
        print("1. Training mode (automated training)")
        print("2. Interactive mode (training with feedback)")
        print("3. Deployment mode (real-time inference only)")
        mode = input("Enter choice (1, 2, or 3): ").strip()
        
        if mode == "3":
            mode = "deploy"
    
    # Set episodes based on mode
    if mode == "deploy":
        episodes = 0  # Real-time only
        print("\n🚀 Deployment Mode")
        print("Running in real-time inference mode (no training).")
        print("Use interactive_inference.py for interactive testing.\n")
    else:
        # Training modes
        if mode == "2":
            print("\n📝 Interactive Training Mode")
            print("You will be asked for feedback on each prompt optimization.")
            print("Enter 'y' for satisfied, 'n' for not satisfied, or 'skip' to skip feedback.\n")
        else:
            print("\n🤖 Automated Training Mode")
            print("Training will proceed without user feedback.\n")
        
        # Get number of episodes
        try:
            episodes_input = input("Enter number of training episodes (default: 10): ").strip()
            episodes = int(episodes_input) if episodes_input else 10
        except ValueError:
            print("Invalid input. Using default 10 episodes.")
            episodes = 10

    # Load pre-trained model if available
    model_path = "saved_model/ppo_actor.pth"
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"✅ Loaded pre-trained model from {model_path}")
    else:
        print("⚠️  No pre-trained model found. Starting with untrained model.")

    # Main execution loop
    if episodes == 0:
        # Deployment mode - just run inference
        print("🎯 Running in deployment mode (no training)")
        print("Use interactive_inference.py for interactive testing.")
        return
    else:
        # Training mode
        for episode in range(episodes):
            total_reward = 0
            print(f"\n🎯 Episode {episode+1}/{episodes}")
            print("-" * 30)
            
            for i, prompt in enumerate(env.prompts):
                print(f"\n📝 Processing prompt {i+1}/{len(env.prompts)}")
                env.original_prompt = prompt
                state = env.encode(prompt).unsqueeze(0).to(device)
                
                # PPO action selection returns action, log_prob, and value
                action, log_prob, value = agent.select_action(state)
                action = torch.tensor(action, dtype=torch.float32).to(device)
                log_prob = log_prob.to(device)
                value = value.to(device)
                
                refined_prompt = env.decode(action.squeeze())
                response = env.real_llm_response(refined_prompt)
                
                # Get user feedback if in interactive mode
                user_satisfied = None
                if mode == "2":
                    user_satisfied = env.get_user_feedback(prompt, refined_prompt, response)
                
                # Calculate reward with or without feedback
                reward = env.calculate_reward_with_feedback(prompt, refined_prompt, response, user_satisfied)

                next_state = env.encode(refined_prompt).unsqueeze(0).to(device)
                done = True

                # PPO training step with additional parameters
                agent.train_step(state, action.unsqueeze(0), reward, log_prob, value, done)
                total_reward += reward

            print(f"\n✅ Episode {episode+1} Complete: Total Reward = {total_reward:.2f}")
            
            # Show feedback statistics if available
            if mode == "2" and env.user_feedback_history:
                stats = env.get_feedback_statistics()
                if isinstance(stats, dict):
                    print(f"📊 Feedback Stats: {stats['satisfaction_rate']:.1f}% satisfaction rate")

        # Save model
        os.makedirs("saved_model", exist_ok=True)
        agent.save("saved_model/ppo_actor.pth")
        print("\n✅ Model saved at saved_model/ppo_actor.pth")
        
        # Final feedback summary
        if mode == "2" and env.user_feedback_history:
            stats = env.get_feedback_statistics()
            if isinstance(stats, dict):
                print(f"\n📈 Training Summary:")
                print(f"   Total feedback collected: {stats['total_feedback']}")
                print(f"   User satisfaction rate: {stats['satisfaction_rate']:.1f}%")
                print(f"   Satisfied responses: {stats['satisfied_count']}/{stats['total_feedback']}")

if __name__ == "__main__":
    main() 