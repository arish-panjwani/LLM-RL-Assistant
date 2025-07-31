from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from utils import PromptEnvironment
from model import DDPGAgent
import torch
import os

# ✅ Import metric & reward logic
from metrics import compute_cosine_similarity, compute_sentiment_score, normalize_user_rating
from reward_functions import clarity_consistency_reward, relevance_reward, hallucination_penalty_reward

app = Flask(__name__)

# Setup
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
    user_feedback_raw = data.get("feedback", "y")

    if not user_prompt:
        return jsonify({"error": "Missing 'prompt' field in JSON"}), 400

    feedback_rating = "up" if user_feedback_raw.lower() == "y" else "down"

    # Encode state and predict
    env.original_prompt = user_prompt
    state = env.encode(user_prompt).unsqueeze(0).to(device)
    action = torch.tensor(agent.select_action(state), dtype=torch.float32).squeeze()
    refined_prompt = env.decode(action)

    # LLM Response
    response = env.real_llm_response(refined_prompt)

    # Metrics + reward
    cosine_sim = compute_cosine_similarity(user_prompt, refined_prompt)
    sentiment_score = compute_sentiment_score(response)
    user_rating = normalize_user_rating(feedback_rating)
    lexical_redundancy = len(refined_prompt.split()) - len(set(refined_prompt.split()))
    halluc_score = env.detect_hallucination(response)
    engagement = len(response.split())
    clarity_score = cosine_sim

    clarity = clarity_consistency_reward(cosine_sim, lexical_redundancy, clarity_score)
    relevance = relevance_reward(user_rating, sentiment_score, engagement)
    final_reward = hallucination_penalty_reward(relevance, halluc_score)

    # Train agent
    reward = final_reward
    next_state = env.encode(refined_prompt).unsqueeze(0).to(device)
    agent.train_step(state, action.unsqueeze(0), reward, next_state, True)
    agent.save("saved_model/ddpg_actor.pth")

    return jsonify({
        "original_prompt": user_prompt,
        "refined_prompt": refined_prompt,
        "llm_response": response,
        "cosine_similarity": round(cosine_sim, 4),
        "sentiment_score": round(sentiment_score, 4),
        "user_rating": user_rating,
        "lexical_redundancy": lexical_redundancy,
        "hallucination_score": round(halluc_score, 4),
        "clarity_reward": round(clarity, 4),
        "relevance_reward": round(relevance, 4),
        "final_reward": round(final_reward, 4)
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)