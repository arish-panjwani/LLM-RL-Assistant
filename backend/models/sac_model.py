from .rl_base import RLModelInterface
import torch.nn as nn
import torch


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = torch.tanh(self.mean(x))  # limit range to [-1, 1]
        return mean  # we only use mean for deterministic actions

class SACAgent:
    def __init__(self, state_dim, action_dim, device):
        self.device = device
        self.actor = ActorNetwork(state_dim, action_dim).to(device)

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()
        return action

    def load(self, path):
        self.actor.load_state_dict(torch.load(path, map_location=self.device))
        self.actor.eval()

class SACModelInterface:
    def __init__(self, model_path):
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.env = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dim = self.encoder.get_sentence_embedding_dimension()
        action_dim = state_dim
        self.model = SACAgent(state_dim, action_dim, self.device)
        self.model.load(model_path)

    def optimize_prompt(self, prompt):
        self.env.original_prompt = prompt
        state = self.env.encode(prompt).unsqueeze(0).to(self.device)
        action = torch.tensor(self.model.select_action(state.cpu().numpy()), dtype=torch.float32)
        return action

    def decode_and_evaluate(self, action):
        refined_prompt = self.env.decode(action)
        response = self.env.real_llm_response(refined_prompt)
        return refined_prompt, response


class SACModel(RLModelInterface):

    def __init__(self, model_path):
        self.model =  SACModelInterface(model_path)

    def generate_response(self, text: str) -> str:
        return self.model.optimize_prompt(text)

    def feedback(self, text: str, liked: bool) -> bool:
        score = 2 if liked else -1
        #self.model.feedback(text)
        return True
