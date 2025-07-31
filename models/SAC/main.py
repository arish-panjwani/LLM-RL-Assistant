import torch
from sentence_transformers import SentenceTransformer
from utils import PromptRLWrapper
from stable_baselines3 import SAC
import os

def main():
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    env = PromptRLWrapper(encoder)

    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./logs", buffer_size=100_000)

    model.learn(total_timesteps=1000)  # Adjust as needed

    os.makedirs("saved_model", exist_ok=True)
    model.save("saved_model/sac_sb3_model")
    print("✅ Model saved to saved_model/sac_sb3_model.zip")

if __name__ == "__main__":
    main()
