import torch
from sentence_transformers import SentenceTransformer
from utils import PromptEnvironment
from promodel import SACAgent
from metrics import compute_cosine_similarity, compute_sentiment_score, compute_hallucination_penalty
from reward_functions import clarity_score, normalize_user_rating

class SACModelInterface:
    def __init__(self, model_path):
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.env = PromptEnvironment(self.encoder)
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

if __name__ == "__main__":
    model = SACModelInterface("saved_model/sac_policy.pth")
    user_prompt = input("🧠 Enter your prompt: ")

    # Optimize prompt
    action = model.optimize_prompt(user_prompt)
    print("Optimized:", action.tolist())

    # Decode and get LLM response
    refined_prompt, response = model.decode_and_evaluate(action)
    print("\n✨ Refined Prompt:", refined_prompt)
    print("💬 LLM Response:", response)

    feedback = input("👍 Was the response helpful? (y/n): ").strip().lower()
    rating = 1 if feedback == "y" else -1 if feedback == "n" else 0

    # 📊 Compute metrics
    cosine = compute_cosine_similarity(user_prompt, refined_prompt)
    sentiment = compute_sentiment_score(response)
    hallucination = compute_hallucination_penalty(user_prompt, response)
    clarity = clarity_score(refined_prompt)
    normalized = normalize_user_rating(rating)

    reward = normalized + sentiment - hallucination + clarity

    print("\n📈 Evaluation Metrics:")
    print(f"🔹 Cosine Similarity       : {cosine:.4f}")
    print(f"🔹 Sentiment Score         : {sentiment:.4f}")
    print(f"🔹 Hallucination Penalty   : {hallucination:.4f}")
    print(f"🔹 Clarity Score (Groq)    : {clarity:.4f}")
    print(f"🔹 Normalized User Rating  : {normalized:.2f}")
    print(f"✅ Final Reward            : {reward:.2f}")