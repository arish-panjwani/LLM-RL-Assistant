# 📦 main.py

import torch
from sentence_transformers import SentenceTransformer
from utils import PromptEnvironment
from model import DDPGAgent
import os

def main():
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    env = PromptEnvironment(encoder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim = encoder.get_sentence_embedding_dimension()
    action_dim = state_dim

    agent = DDPGAgent(state_dim=state_dim, action_dim=action_dim, device=device)

    episodes = 1000  # Use > 0 for training; use 0 for real-time only
    for episode in range(episodes):
        total_reward = 0
        for prompt in env.prompts:
            env.original_prompt = prompt  # Set original prompt
            state = env.encode(prompt).unsqueeze(0).to(device)
            action = torch.tensor(agent.select_action(state), dtype=torch.float32).to(device)
            # action = torch.tensor(agent.select_action(state), dtype=torch.float32).unsqueeze(0).to(device)
            refined_prompt = env.decode(action.squeeze())
            response = env.real_llm_response(refined_prompt)
            reward = env.calculate_reward(prompt, refined_prompt, response)
            reward = float(reward) 

            next_state = env.encode(refined_prompt).unsqueeze(0).to(device)
            done = True

            # agent.train_step(state, action.unsqueeze(0), reward, next_state, done)
            agent.train_step(state, action, reward, next_state, done)
            total_reward += reward

        print(f"Episode {episode+1}: Total Reward = {total_reward:.2f}")

    os.makedirs("saved_model", exist_ok=True)
    agent.save("saved_model/ddpg_actor.pth")
    print("✅ Model saved at saved_model/ddpg_actor.pth")

if __name__ == "__main__":
    main()

