import os
import re
import random
import torch
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables (e.g., GROQ_API_KEY, GROQ_API_BASE)
load_dotenv()

# ✅ Import your reward and metric logic
from metrics import compute_cosine_similarity, compute_sentiment_score, normalize_user_rating
from reward_functions import clarity_consistency_reward, relevance_reward, hallucination_penalty_reward

nltk.download('vader_lexicon')


class PromptEnvironment:
    def __init__(self, encoder):
        self.encoder = encoder
        self.sid = SentimentIntensityAnalyzer()
        self.client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url=os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")
        )
        self.prompts = [
            "How do I cook rice?",
            "Tell me something about Mars.",
            "What are the symptoms of flu?",
            "Suggest a healthy lunch.",
            "How to learn Python quickly?"
        ]
        self.original_prompt = None

    def reset(self):
        self.original_prompt = random.choice(self.prompts)
        return self.encode(self.original_prompt)

    def encode(self, text):
        emb = self.encoder.encode(text)
        return torch.tensor(emb, dtype=torch.float32)

    def decode(self, embedding):
        # Placeholder decode logic (for embedding-to-text)
        return self.original_prompt or "Can you rephrase this prompt to be clearer?"

    def real_llm_response(self, prompt):
        try:
            response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": "Rewrite the prompt clearly for an LLM."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=100
            )
            full_text = response.choices[0].message.content.strip()
            cleaned = re.sub(r'^.*?"(.*?)"\s*$', r'\1', full_text, flags=re.DOTALL)
            return cleaned.strip('"').strip()
        except Exception as e:
            return f"Error during LLM response: {str(e)}"

    def detect_hallucination(self, response: str) -> float:
        hallucination_phrases = [
            "I'm not sure", "might be", "could be", "possibly",
            "not certain", "I think", "perhaps"
        ]
        return sum(phrase in response.lower() for phrase in hallucination_phrases) / len(hallucination_phrases)

    def compute_metrics(self, original_prompt, refined_prompt, response, user_feedback="n"):
        cosine_sim = compute_cosine_similarity(original_prompt, refined_prompt)
        sentiment_score = compute_sentiment_score(response)
        user_rating = normalize_user_rating(user_feedback)
        lexical_redundancy = len(refined_prompt.split()) - len(set(refined_prompt.split()))
        halluc_score = self.detect_hallucination(response)
        engagement = len(response.split())
        clarity_score = cosine_sim  # simplified proxy

        clarity = clarity_consistency_reward(cosine_sim, lexical_redundancy, clarity_score)
        relevance = relevance_reward(user_rating, sentiment_score, engagement)
        final_reward = hallucination_penalty_reward(relevance, halluc_score)

        return (
            cosine_sim,
            sentiment_score,
            halluc_score,
            lexical_redundancy,
            clarity,
            user_rating,
            final_reward
        )


# ✅ Pretrained sentence encoder loader
def load_pretrained_encoder():
    return SentenceTransformer('all-MiniLM-L6-v2')