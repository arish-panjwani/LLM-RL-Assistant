# main.py

import os
import gym
import torch
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from sentence_transformers import SentenceTransformer
from utils import PromptEnvironment

class PromptGym(gym.Env):
    def __init__(self, encoder):
        super().__init__()
        self.env = PromptEnvironment(encoder)
        self.encoder = encoder
        self.prompts = self.env.prompts
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(encoder.get_sentence_embedding_dimension(),), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(encoder.get_sentence_embedding_dimension(),), dtype=np.float32)
        self.current_index = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reset(self):
        self.env.original_prompt = self.prompts[self.current_index % len(self.prompts)]
        self.current_index += 1
        return self.env.encode(self.env.original_prompt).numpy()

    def step(self, action):
        state = self.env.encode(self.env.original_prompt).unsqueeze(0).to(self.device)
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(self.device)
        refined_prompt = self.env.decode(action_tensor.squeeze())
        response = self.env.real_llm_response(refined_prompt)
        reward = float(self.env.calculate_reward(self.env.original_prompt, refined_prompt, response))
        obs = self.env.encode(refined_prompt).numpy()
        done = True  # 1 step per prompt
        return obs, reward, done, {}

def main():
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    env = PromptGym(encoder)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = DDPG("MlpPolicy", env, verbose=1, action_noise=action_noise, tensorboard_log="./ddpg_logs")

    model.learn(total_timesteps=1000)
    
    os.makedirs("saved_model", exist_ok=True)
    model.save("saved_model/ddpg_agent.zip")
    print("✅ Model saved to saved_model/ddpg_agent.zip")

if __name__ == "__main__":
    main()
