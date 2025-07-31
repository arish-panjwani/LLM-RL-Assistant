from .rl_base import RLModelInterface

import torch
from sentence_transformers import SentenceTransformer
import zipfile
import pickle
import tempfile
import os

class Actor(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, action_dim)
        self.log_std = torch.nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = torch.nn.functional.relu(self.fc1(state))
        x = torch.nn.functional.relu(self.fc2(x))
        mean = torch.tanh(self.fc3(x))
        std = torch.exp(self.log_std)
        return mean, std

class PPOModel:
    def __init__(self, model_path="saved_model/ppo_model.zip"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.load_model(model_path)
    
    def load_model(self, model_path):
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(model_path, 'r') as zipf:
                zipf.extractall(temp_dir)
            
            model_data_path = os.path.join(temp_dir, 'model_data.pkl')
            with open(model_data_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.actor = Actor(model_data['state_dim'], model_data['action_dim']).to(self.device)
            self.actor.load_state_dict(model_data['actor_state_dict'])
            self.actor.eval()
    
    def predict(self, text: str) -> str:
        state = self.encoder.encode(text, convert_to_tensor=True).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mean, std = self.actor(state)
            action = torch.tanh(mean)
        
        return f"Optimized: {action.squeeze().cpu().numpy()[:50].tolist()}"
    
    def feedback(self, text: str):
        print(f"Feedback received for: {text}")
        return True

# Usage
#model = PPOModel("C:\Users\Dell\LLM-RL-Assistant\backend\model_files\ppo_model.zip")
#response = model.predict("Mueez Type your test here")
#print(response)

class PPOModelHandler(RLModelInterface):
    def __init__(self, pth_file_path, model_class = None):
        self.model = PPOModel(pth_file_path)

    def generate_response(self, text: str) -> str:
        return self.model.predict(text)
        #return text #by pass

    def feedback(self, text: str, liked: bool) -> bool:
        score = 2 if liked else -1
        #self.model.feedback(text)
        return True
