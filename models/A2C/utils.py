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
        
        # Get API keys from environment variables with enhanced validation
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            print("❌ GROQ_API_KEY not found in environment variables.")
            print("📝 Please create a .env file in the project root with:")
            print("   GROQ_API_KEY=your_groq_api_key_here")
            print("   WOLFRAM_APP_ID=your_wolfram_app_id_here (optional)")
            print("   GOOGLE_API_KEY=your_google_api_key_here (optional)")
            raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")
        
        # Validate API key format (basic check)
        if len(groq_api_key) < 10:
            raise ValueError("GROQ_API_KEY appears to be invalid (too short). Please check your .env file.")
            
        self.client = OpenAI(
            api_key=groq_api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        
        # Optional API keys
        self.wolfram_app_id = os.getenv("WOLFRAM_ALPHA_APP_ID")  # Your specific variable name
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.google_cse_id = os.getenv("GOOGLE_CSE_ID")  # Your Google Custom Search Engine ID
        
        # Log successful initialization
        print("✅ Environment variables loaded successfully")
        print(f"   Groq API: {'✅ Configured' if groq_api_key else '❌ Missing'}")
        print(f"   Wolfram Alpha API: {'✅ Configured' if self.wolfram_app_id else '⚠️  Optional'}")
        print(f"   Google API: {'✅ Configured' if self.google_api_key else '⚠️  Optional'}")
        print(f"   Google CSE ID: {'✅ Configured' if self.google_cse_id else '⚠️  Optional'}")
        
        # Show your 4 specific API keys
        print("\n🔍 Your configured API keys:")
        print(f"   GROQ_API_KEY: {'✅ Set' if groq_api_key else '❌ Not set'}")
        print(f"   WOLFRAM_ALPHA_APP_ID: {'✅ Set' if self.wolfram_app_id else '❌ Not set'}")
        print(f"   GOOGLE_API_KEY: {'✅ Set' if self.google_api_key else '❌ Not set'}")
        print(f"   GOOGLE_CSE_ID: {'✅ Set' if self.google_cse_id else '❌ Not set'}")
        
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

        # Enhanced reward formula for A2C
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

    def test_api_connection(self):
        """Test the Groq API connection"""
        try:
            response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "user", "content": "Hello, this is a test message."}
                ],
                temperature=0.1,
                max_tokens=10
            )
            return True, "API connection successful"
        except Exception as e:
            return False, f"API connection failed: {str(e)}"

    def test_wolfram_connection(self):
        """Test the Wolfram Alpha API connection"""
        if not self.wolfram_app_id:
            return False, "Wolfram API key not configured"
        
        try:
            # Test with a simple query
            test_query = "2+2"
            url = "http://api.wolframalpha.com/v1/result"
            params = {
                "appid": self.wolfram_app_id,
                "i": test_query,
                "units": "metric"
            }
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return True, f"Wolfram API connection successful: {response.text}"
            else:
                return False, f"Wolfram API error: HTTP {response.status_code}"
        except Exception as e:
            return False, f"Wolfram API connection failed: {str(e)}"

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
            error_msg = str(e)
            if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
                return f"❌ API Authentication Error: Please check your GROQ_API_KEY in .env file"
            elif "quota" in error_msg.lower() or "rate" in error_msg.lower():
                return f"⚠️  API Rate Limit/Quota Error: {error_msg}"
            else:
                return f"❌ API Error: {error_msg}"

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
        if not self.wolfram_app_id:
            return "Wolfram Alpha API not configured"
            
        try:
            url = "http://api.wolframalpha.com/v1/result"
            params = {
                "appid": self.wolfram_app_id,
                "i": query,
                "units": "metric"
            }
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.text
            else:
                return f"Wolfram query failed: {response.status_code}"
        except Exception as e:
            return f"Wolfram API error: {str(e)}"

    def google_search(self, query):
        """Perform a Google search using Custom Search Engine"""
        if not self.google_api_key or not self.google_cse_id:
            return "Google Search API not configured (needs both API key and CSE ID)"
            
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.google_api_key,
                "cx": self.google_cse_id,
                "q": query,
                "num": 3  # Get top 3 results
            }
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if "items" in data:
                    results = []
                    for item in data["items"][:3]:
                        results.append(f"• {item['title']}: {item['snippet']}")
                    return "\n".join(results)
                else:
                    return "No search results found"
            else:
                return f"Google search failed: {response.status_code}"
        except Exception as e:
            return f"Google search error: {str(e)}"


def load_pretrained_encoder():
    return SentenceTransformer('all-MiniLM-L6-v2') 