# server/api_routes.py
from flask import request, jsonify
from utils.groq_client import GroqClient
from environment.reward_calculator import RewardCalculator
from config.config import config
from models.ppo.ppo_model import PPOModel

def register_routes(app):
    groq_client = GroqClient(config.GROQ_API_KEY)
    reward_calc = RewardCalculator(groq_client, config)
    model = PPOModel(config)

    @app.route("/optimize", methods=["POST"])
    def optimize():
        data = request.get_json()
        prompt = data.get("prompt", "")
        embedding = reward_calc.embedding_model.encode([prompt])[0]
        optimized_embedding = model.optimize_prompt(embedding)
        new_prompt = f"Please provide a clear and specific answer to: {prompt}"
        response = groq_client.get_response(new_prompt)

        clarity = reward_calc.calculate_clarity_reward(prompt, new_prompt)
        relevance = reward_calc.calculate_relevance_reward(prompt, response)
        hallucination = reward_calc.calculate_hallucination_penalty(response)

        return jsonify({
            "success": True,
            "data": {
                "original_prompt": prompt,
                "optimized_prompt": new_prompt,
                "response": response,
                "model_used": "PPO",
                "metrics": {
                    "clarity_score": clarity,
                    "relevance_score": relevance,
                    "hallucination_penalty": hallucination,
                    "total_reward": clarity + relevance - hallucination
                }
            }
        })

    @app.route("/compare", methods=["POST"])
    def compare():
        return jsonify({"message": "Comparison not implemented yet"})

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})
