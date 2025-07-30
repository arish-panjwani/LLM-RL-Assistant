# 🔧 utils.py

import random
import torch
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os
import re

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
        if self.original_prompt:
            return self.original_prompt
        else:
            return "Can you rephrase this prompt to be clearer?"

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

            # Clean typical prefixes like "Here is..." or markdown quotes
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

    def calculate_reward(self, original_prompt, refined_prompt, response):
        orig_vec = self.encoder.encode(original_prompt)
        ref_vec = self.encoder.encode(refined_prompt)
        cosine_sim = float(util.cos_sim(torch.tensor(orig_vec), torch.tensor(ref_vec)))
        clarity_rating = random.uniform(5, 10)
        words = refined_prompt.lower().split()
        redundancy_penalty = len(words) - len(set(words))
        sentiment_score = self.sid.polarity_scores(response)['compound']
        hallucination_penalty = self.detect_hallucination(response)

        # Enhanced reward formula
        λ1, λ2, λ3, γ = 1.0, 0.5, 1.0, 2.0
        return (
            λ1 * cosine_sim
            - λ2 * redundancy_penalty
            + λ3 * clarity_rating
            + 0.5 * sentiment_score
            - γ * hallucination_penalty
        )


def load_pretrained_encoder():
    return SentenceTransformer('all-MiniLM-L6-v2')
