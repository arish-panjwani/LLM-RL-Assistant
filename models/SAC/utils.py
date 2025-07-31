import os
import random
import torch
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import gym
from gym import spaces

# ✅ NEW: Import your metric & reward functions
from metrics import compute_cosine_similarity, compute_sentiment_score, normalize_user_rating
from reward_functions import clarity_consistency_reward, relevance_reward, hallucination_penalty_reward

nltk.download('vader_lexicon')
load_dotenv()

class PromptEnvironment:
    def __init__(self, encoder):
        self.encoder = encoder
        self.sid = SentimentIntensityAnalyzer()
        self.client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url=os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")
        )
        self.prompts = [
            "Explain quantum computing in simple terms.",
            "What is the future of AI in healthcare?",
            "Give tips for improving public speaking skills.",
            "Suggest some eco-friendly lifestyle habits.",
            "What are key features of a successful startup?"
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
            return f'Can you rephrase the following prompt to be clearer and more specific for an LLM: "{self.original_prompt}"'
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
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error during LLM response: {str(e)}"

    def detect_hallucination(self, response: str) -> float:
        hallucination_phrases = [
            "I'm not sure", "might be", "could be", "possibly",
            "not certain", "I think", "perhaps"
        ]
        return sum(phrase in response.lower() for phrase in hallucination_phrases) / len(hallucination_phrases)

    # ✅ UPDATED REWARD FUNCTION
    def calculate_reward(self, original_prompt, refined_prompt, response, user_feedback="Good", user_rating_raw="up", groq_clarity_score=0.0):
        cosine_sim = compute_cosine_similarity(original_prompt, refined_prompt)
        sentiment_score = compute_sentiment_score(user_feedback)
        user_rating = normalize_user_rating(user_rating_raw)
        lexical_redundancy = len(refined_prompt.split()) - len(set(refined_prompt.split()))
        hallucination_score = self.detect_hallucination(response)
        engagement_length = len(user_feedback.split())

        clarity = clarity_consistency_reward(cosine_sim, lexical_redundancy, groq_clarity_score)
        relevance = relevance_reward(user_rating, sentiment_score, engagement_length)
        final_reward = hallucination_penalty_reward(relevance, hallucination_score)

        return final_reward

def load_pretrained_encoder():
    return SentenceTransformer('all-MiniLM-L6-v2')

class PromptRLWrapper(gym.Env):
    def __init__(self, encoder):
        super().__init__()
        self.env = PromptEnvironment(encoder)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(encoder.get_sentence_embedding_dimension(),),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(encoder.get_sentence_embedding_dimension(),),
            dtype=np.float32
        )

    def reset(self):
        return self.env.reset().numpy()

    def step(self, action):
        action = torch.tensor(action, dtype=torch.float32)
        refined = self.env.decode(action)
        response = self.env.real_llm_response(refined)
        reward = self.env.calculate_reward(self.env.original_prompt, refined, response)
        next_state = self.env.encode(refined).numpy()
        done = True
        return next_state, reward, done, {}