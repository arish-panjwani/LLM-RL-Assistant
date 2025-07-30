from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from utils import PromptEnvironment
from model import DDPGAgent
import torch

app = Flask(__name__)

# Load encoder, environment, and model
encoder = SentenceTransformer("all-MiniLM-L6-v2")
env = PromptEnvironment(encoder)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_dim = encoder.get_sentence_embedding_dimension()
action_dim = state_dim

agent = DDPGAgent(state_dim=state_dim, action_dim=action_dim, device=device)
agent.load("saved_model/ddpg_actor.pth")
print("✅ Loaded DDPG model for API.")

@app.route("/refine", methods=["POST"])
def refine_prompt():
    data = request.get_json()
    user_prompt = data.get("prompt")

    if not user_prompt:
        return jsonify({"error": "Missing 'prompt' field in JSON"}), 400

    # Set the prompt for context in decode
    env.original_prompt = user_prompt

    # Encode and predict
    state = env.encode(user_prompt).unsqueeze(0).to(device)
    action = torch.tensor(agent.select_action(state), dtype=torch.float32).squeeze()
    refined_prompt = env.decode(action)

    # Get LLM response
    response = env.real_llm_response(refined_prompt)

    return jsonify({
        "original_prompt": user_prompt,
        "refined_prompt": refined_prompt,
        "llm_response": response
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
