from .rl_base import RLModelInterface

import torch
from sentence_transformers import SentenceTransformer
from utils import PromptEnvironment
from model import DDPGAgent

class DDPGModelInterface:
    def __init__(self, model_path):
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.env = PromptEnvironment(self.encoder)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        state_dim = self.encoder.get_sentence_embedding_dimension()
        action_dim = state_dim

        self.model = DDPGAgent(state_dim, action_dim, self.device)
        self.model.load(model_path)
        print("✅ DDPG model loaded.")

    def optimize_prompt(self, prompt):
        self.env.original_prompt = prompt
        state = self.env.encode(prompt).unsqueeze(0).to(self.device)
        action = torch.tensor(self.model.select_action(state.cpu().numpy()), dtype=torch.float32).squeeze()
        refined_prompt = self.env.decode(action)
        response = self.env.real_llm_response(refined_prompt)
        return refined_prompt, response, action, state


class DDPGModel(RLModelInterface):

    def __init__(self, model_path):
        self.model = DDPGModelInterface(model_path)

    def generate_response(self, text: str) -> str:
        return self.model.optimize_prompt(text)

    def feedback(self, text: str, liked: bool) -> bool:
        return self.model.feedback(text, liked)