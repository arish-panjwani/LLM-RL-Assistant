from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from model import SACAgent
from utils import PromptEnvironment
import torch
import os

app = Flask(__name__)

encoder = SentenceTransformer("all-MiniLM-L6-v2")
env = PromptEnvironment(encoder)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dim = encoder.get_sentence_embedding_dimension()
action_dim = state_dim

agent = SACAgent(state_dim, action_dim, device)

model_path = "saved_model/sac_policy.pth"
if os.path.exists(model_path):
    try:
        agent.load(model_path)
        print("✅ SAC model loaded.")
    except Exception as e:
        print(f"❌ Error loading SAC model: {e}")
else:
    print("⚠️ Model file not found. A new model will be used.")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    prompt = data.get("prompt", "")

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    user_feedback = data.get("user_feedback", "Good")
    user_rating_raw = data.get("user_rating", "up")
    groq_clarity_score = float(data.get("groq_clarity_score", 0.0))

    env.original_prompt = prompt
    state = env.encode(prompt).unsqueeze(0).to(device)

    action = torch.tensor(agent.select_action(state.cpu().numpy()), dtype=torch.float32).squeeze()
    refined_prompt = env.decode(action)
    response = env.real_llm_response(refined_prompt)

    reward = env.calculate_reward(prompt, refined_prompt, response, user_feedback, user_rating_raw, groq_clarity_score)

    next_state = env.encode(refined_prompt).unsqueeze(0).to(device)
    done = True
    agent.store_transition(state.cpu().numpy(), action.cpu().numpy(), reward, next_state.cpu().numpy(), done)
    agent.train_step()
    os.makedirs("saved_model", exist_ok=True)
    agent.save(model_path)

    print(f"\n📨 Original Prompt: {prompt}")
    print(f"📝 Refined Prompt: {refined_prompt}")
    print(f"💬 LLM Response: {response}")
    print(f"🏆 Final Reward: {reward:.2f}")

    return jsonify({
        "original_prompt": prompt,
        "refined_prompt": refined_prompt,
        "llm_response": response,
        "reward": reward
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)