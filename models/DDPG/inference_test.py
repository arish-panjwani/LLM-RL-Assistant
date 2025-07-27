import torch
from sentence_transformers import SentenceTransformer
from utils import PromptEnvironment
from model import DDPGAgent

def main():
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    env = PromptEnvironment(encoder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim = encoder.get_sentence_embedding_dimension()
    action_dim = state_dim

    agent = DDPGAgent(state_dim=state_dim, action_dim=action_dim, device=device)
    agent.load("saved_model/ddpg_actor.pth")
    print("Loaded trained DDPG model.")

    test_prompt = input("🔍 Enter your prompt: ")
    env.original_prompt = test_prompt  # Set for decode logic

    state = env.encode(test_prompt).unsqueeze(0).to(device)
    action = torch.tensor(agent.select_action(state), dtype=torch.float32).squeeze()
    refined_prompt = env.decode(action)
    response = env.real_llm_response(refined_prompt)

    print(f"\n✨ Refined Prompt: {refined_prompt}")
    print(f"💬 LLM Response: {response}")

    feedback = input("👍 Was the response helpful? (y/n): ").strip().lower()
    rating = 1 if feedback == "y" else -1 if feedback == "n" else 0
    sentiment_score = env.sid.polarity_scores(response)['compound']
    reward = 1.0 * rating + 0.5 * sentiment_score

    next_state = env.encode(refined_prompt).unsqueeze(0).to(device)
    done = True

    agent.train_step(state, action.unsqueeze(0), reward, next_state, done)
    agent.save("saved_model/ddpg_actor.pth")
    print(f"Updated model with reward: {reward:.2f}")

if __name__ == "__main__":
    main()
