import random
import torch
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

nltk.download('vader_lexicon')

class PromptEnvironment:
    def __init__(self, encoder):
        self.encoder = encoder
        self.sid = SentimentIntensityAnalyzer()
        
        # Get API keys from environment variables
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")
            
        self.client = OpenAI(
            api_key=groq_api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        self.wolfram_app_id = os.getenv("WOLFRAM_APP_ID")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        
        # Training prompts (can be extended)
        self.prompts = [
            "How do I cook rice?",
            "Tell me something about Mars.",
            "What are the symptoms of flu?",
            "Suggest a healthy lunch.",
            "How to learn Python quickly?"
        ]
        
        # User feedback storage
        self.user_feedback_history = []
        self.original_prompt = None
        
        # Dynamic prompt templates (not hardcoded)
        self.prompt_templates = [
            "Rewrite this prompt to be more specific: {prompt}",
            "Make this prompt clearer and more detailed: {prompt}",
            "Optimize this prompt for better AI understanding: {prompt}",
            "Enhance this prompt with more context: {prompt}",
            "Refine this prompt for improved clarity: {prompt}"
        ]

    def reset(self):
        self.original_prompt = random.choice(self.prompts)
        return self.encode(self.original_prompt)

    def encode(self, text):
        emb = self.encoder.encode(text)
        return torch.tensor(emb, dtype=torch.float32)

    def decode(self, embedding):
        """Dynamic prompt generation based on embedding"""
        if self.original_prompt:
            # Use embedding to select template dynamically
            template_idx = int(abs(hash(str(embedding.tolist()))) % len(self.prompt_templates))
            template = self.prompt_templates[template_idx]
            return template.format(prompt=self.original_prompt)
        else:
            return "Please provide a prompt to optimize."

    def get_user_feedback(self, original_prompt, refined_prompt, llm_response):
        """Get user feedback on the refined prompt"""
        print(f"\n📝 Original Prompt: {original_prompt}")
        print(f"🔄 Refined Prompt: {refined_prompt}")
        print(f"🤖 LLM Response: {llm_response}")
        
        while True:
            feedback = input("\n❓ Are you satisfied with this optimization? (y/n/skip): ").strip().lower()
            if feedback in ['y', 'n', 'skip']:
                break
            print("Please enter 'y' for yes, 'n' for no, or 'skip' to skip feedback")
        
        if feedback != 'skip':
            # Store feedback for learning
            self.user_feedback_history.append({
                'original': original_prompt,
                'refined': refined_prompt,
                'response': llm_response,
                'satisfied': feedback == 'y',
                'timestamp': len(self.user_feedback_history)
            })
            
            return feedback == 'y'
        return None  # Skip feedback

    def calculate_reward_with_feedback(self, original_prompt, refined_prompt, response, user_satisfied=None):
        """Calculate reward including user feedback"""
        # Base reward calculation
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
        base_reward = (
            λ1 * cosine_sim
            - λ2 * redundancy_penalty
            + λ3 * clarity_rating
            + 0.5 * sentiment_score
            - γ * hallucination_penalty
        )
        
        # Add user feedback bonus/penalty
        if user_satisfied is not None:
            feedback_bonus = 2.0 if user_satisfied else -1.0
            return base_reward + feedback_bonus
        
        return base_reward

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

    def calculate_reward(self, original_prompt, refined_prompt, response):
        """Legacy reward function (for backward compatibility)"""
        return self.calculate_reward_with_feedback(original_prompt, refined_prompt, response)

    def get_feedback_statistics(self):
        """Get statistics about user feedback"""
        if not self.user_feedback_history:
            return "No feedback collected yet."
        
        total_feedback = len(self.user_feedback_history)
        satisfied_count = sum(1 for f in self.user_feedback_history if f['satisfied'])
        satisfaction_rate = satisfied_count / total_feedback * 100
        
        return {
            'total_feedback': total_feedback,
            'satisfied_count': satisfied_count,
            'satisfaction_rate': satisfaction_rate,
            'recent_feedback': self.user_feedback_history[-5:]  # Last 5 feedback entries
        }

    def wolfram_query(self, query):
        """Query Wolfram Alpha for computational knowledge"""
        try:
            url = "http://api.wolframalpha.com/v1/result"
            params = {
                "appid": self.wolfram_app_id,
                "i": query,
                "units": "metric"
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.text
            else:
                return f"Wolfram query failed: {response.status_code}"
        except Exception as e:
            return f"Wolfram API error: {str(e)}"

    def google_search(self, query):
        """Perform a Google search (simplified version)"""
        try:
            # Note: This is a simplified version. For full Google Search API, you'd need additional setup
            return f"Google search for: {query}"
        except Exception as e:
            return f"Google search error: {str(e)}"


def load_pretrained_encoder():
    return SentenceTransformer('all-MiniLM-L6-v2') 