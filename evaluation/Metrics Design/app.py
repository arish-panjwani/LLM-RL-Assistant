from flask import Flask, request, jsonify
from metrics import compute_cosine_similarity, compute_sentiment_score, normalize_user_rating
from reward_functions import clarity_consistency_reward, relevance_reward, hallucination_penalty_reward

app = Flask(__name__)

@app.route("/metrics", methods=["POST"])
def compute_metrics():
    data = request.get_json()

    response1 = data.get("response1", "")
    response2 = data.get("response2", "")
    user_feedback = data.get("user_feedback", "")
    raw_rating = data.get("user_rating", "neutral")
    lexical_redundancy = data.get("lexical_redundancy", 0.0)
    groq_clarity_score = data.get("groq_clarity_score", 0.0)
    hallucination_score = data.get("hallucination_score", 0.0)

    # Core metrics
    cos_sim = compute_cosine_similarity(response1, response2)
    sentiment = compute_sentiment_score(user_feedback)
    user_rating = normalize_user_rating(raw_rating)
    engagement_length = len(user_feedback.split())

    # Apply individual rewards
    clarity_reward = clarity_consistency_reward(
        cosine_similarity=cos_sim,
        lexical_redundancy=lexical_redundancy,
        groq_clarity_score=groq_clarity_score
    )

    relevance = relevance_reward(
        user_rating=user_rating,
        sentiment_score=sentiment,
        engagement_length=engagement_length
    )

    final_reward = hallucination_penalty_reward(
        base_reward=relevance,
        hallucination_score=hallucination_score
    )

    return jsonify({
        "cosine_similarity": round(cos_sim, 4),
        "sentiment_score": round(sentiment, 4),
        "user_rating": user_rating,
        "engagement_length": engagement_length,
        "clarity_reward": round(clarity_reward, 4),
        "relevance_reward": round(relevance, 4),
        "final_reward": round(final_reward, 4)
    })

if __name__ == "__main__":
    app.run(debug=True)
