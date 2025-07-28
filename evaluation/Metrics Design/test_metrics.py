from metrics import compute_cosine_similarity, compute_sentiment_score, normalize_user_rating, calculate_reward

response1 = "The Eiffel Tower is in Paris, France."
response2 = "Paris has the Eiffel Tower located in France."

user_feedback = "That's exactly what I was looking for!"
raw_rating = "up"

cos_sim = compute_cosine_similarity(response1, response2)
sentiment = compute_sentiment_score(user_feedback)
user_rating = normalize_user_rating(raw_rating)
reward = calculate_reward(cos_sim, sentiment, user_rating)

print("🧠 Cosine Similarity:", round(cos_sim, 4))
print("💬 Sentiment Score:", round(sentiment, 4))
print("👍 Normalized User Rating:", user_rating)
print("🏆 Final Reward:", round(reward, 4))
