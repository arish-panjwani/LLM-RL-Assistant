from .rl_base import RLModelInterface

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import zipfile
import json
import tempfile
import os
import random

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = torch.tanh(self.fc3(x))
        std = torch.exp(self.log_std)
        return mean, std

    def sample_action(self, state):
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        action = normal.sample()
        log_prob = normal.log_prob(action)
        return action, log_prob

class A2CModel:
    def __init__(self, model_path="saved_model/a2c_actor.zip"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.load_model(model_path)
        
        # Your A2C prompt templates
        self.prompt_templates = [
            "Rewrite this prompt to be more specific: {prompt}",
            "Make this prompt clearer and more detailed: {prompt}",
            "Optimize this prompt for better AI understanding: {prompt}",
            "Enhance this prompt with more context: {prompt}",
            "Refine this prompt for improved clarity: {prompt}"
        ]
    
    def load_model(self, model_path):
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(model_path, 'r') as zipf:
                zipf.extractall(temp_dir)
            
            # Load metadata
            metadata_path = os.path.join(temp_dir, 'metadata.json')
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Load actor weights
            actor_path = os.path.join(temp_dir, 'actor.pth')
            actor_weights = torch.load(actor_path, map_location=self.device)
            
            # Create and load actor
            self.actor = Actor(metadata['state_dim'], metadata['action_dim']).to(self.device)
            self.actor.load_state_dict(actor_weights)
            self.actor.eval()
            
            print(f"✅ A2C Model loaded successfully!")
            print(f"   State dimension: {metadata['state_dim']}")
            print(f"   Action dimension: {metadata['action_dim']}")
    
    def decode(self, embedding, original_prompt):
        """Decode embedding to optimized prompt using your templates"""
        if original_prompt:
            # Use embedding to select template dynamically (your method)
            template_idx = int(abs(hash(str(embedding.tolist()))) % len(self.prompt_templates))
            template = self.prompt_templates[template_idx]
            return template.format(prompt=original_prompt)
        else:
            return "Please provide a prompt to optimize."
    
    def predict(self, text: str) -> str:
        """Generate optimized prompt from input text"""
        # Encode the prompt
        state = self.encoder.encode(text, convert_to_tensor=True).unsqueeze(0).to(self.device)
        
        # Get A2C action (optimized prompt)
        with torch.no_grad():
            action, log_prob = self.actor.sample_action(state)
        
        # Decode the action to get optimized prompt
        optimized_prompt = self.decode(action.squeeze(), text)
        
        return optimized_prompt
    
    def feedback(self, text: str, liked: bool):
        """Process feedback (for future training)"""
        print(f"Feedback received for: {text} - {'Liked' if liked else 'Disliked'}")
        return True

"""
# Usage
if __name__ == "__main__":
    model = A2CModel("saved_model/a2c_actor.zip")
    response = model.predict("How do I cook rice?")
    print(f"Original: How do I cook rice?")
    print(f"Optimized: {response}") 
"""

class ATCModel(RLModelInterface):

    def __init__(self, model_path):
        self.model = A2CModel(model_path)

    def generate_response(self, text: str) -> str:
        return self.model.predict(text)

    def feedback(self, text: str, liked: bool) -> bool:
        return self.model.feedback(text, liked)