import torch
from utils import PromptEnvironment, load_pretrained_encoder
from model import DDPGAgent
from metrics import (
    compute_cosine_similarity,
    compute_sentiment_score,
    compute_hallucination_penalty,
    normalize_user_rating
)
from reward_functions import (
    clarity_consistency_reward,
    relevance_reward,
    hallucination_penalty_reward
)

def main():
    # ✅ Load encoder and environment
    encoder = load_pretrained_encoder()
    env = PromptEnvironment(encoder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ Load model
    state_dim = encoder.get_sentence_embedding_dimension()
    action_dim = state_dim
    agent = DDPGAgent(state_dim, action_dim, device)
    agent.load("saved_model/ddpg_actor.pth")
    print("✅ Loaded trained DDPG model.")

    # 🧠 Get user input
    user_prompt = input("🔍 Enter your prompt: ").strip()
    env.original_prompt = user_prompt
    state = env.encode(user_prompt).unsqueeze(0).to(device)
    action = torch.tensor(agent.select_action(state.cpu().numpy()), dtype=torch.float32)

    # ✨ Decode and respond
    refined_prompt = env.decode(action)
    response = env.real_llm_response(refined_prompt)
    print("\n✨ Refined Prompt:", refined_prompt)
    print("💬 LLM Response:", response)

    # ✅ User feedback
    feedback = input("👍 Was the response helpful? (y/n): ").strip().lower()
    rating = 1 if feedback == "y" else -1 if feedback == "n" else 0

    # ✅ Compute Metrics
    cosine = compute_cosine_similarity(user_prompt, refined_prompt)
    sentiment = compute_sentiment_score(response)
    hallucination = compute_hallucination_penalty(user_prompt, response)
    lexical = len(refined_prompt.split()) - len(set(refined_prompt.split()))
    clarity = clarity_consistency_reward(response)
    normalized_rating = normalize_user_rating(rating)
    engagement = len(response.split())
    relevance = relevance_reward(user_prompt, response)
    final_reward = hallucination_penalty_reward(response)

    # 📊 Print metrics
    print("\n📊 Evaluation Metrics:")
    print(f"🔹 Cosine Similarity       : {cosine:.4f}")
    print(f"🔹 Sentiment Score         : {sentiment:.4f}")
    print(f"🔹 Hallucination Penalty   : {hallucination:.4f}")
    print(f"🔹 Lexical Redundancy      : {lexical}")
    print(f"🔹 Clarity Score (Groq)    : {clarity:.4f}")
    print(f"🔹 Normalized User Rating  : {normalized_rating:.2f}")
    print(f"✅ Final Reward            : {final_reward:.4f}")

if __name__ == "__main__":
    main()