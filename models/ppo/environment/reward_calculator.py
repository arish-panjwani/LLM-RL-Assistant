import requests
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from collections import Counter
import numpy as np
import wikipedia
import logging
from utils.groq_client import GroqClient
from typing import Optional
from config.config import Config

logger = logging.getLogger(__name__)

class RewardCalculator:
    def __init__(self, groq_client: GroqClient, config: Config):
        self.groq_client = groq_client
        self.config = config
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def calculate_clarity_reward(self, original_prompt: str, modified_prompt: str) -> float:
        """R1 = λ1·cosine_sim - λ2·redundancy_penalty + λ3·Groq_rating"""
        
        # Generate multiple responses for consistency check
        responses = []
        for _ in range(3):
            resp = self.groq_client.get_response(modified_prompt)
            responses.append(resp)
        
        # Calculate cosine similarity between responses
        embeddings = self.embedding_model.encode(responses)
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                similarities.append(sim)
        
        avg_similarity = np.mean(similarities) if similarities else 0.5
        
        # Calculate redundancy penalty
        words = modified_prompt.lower().split()
        word_counts = Counter(words)
        total_words = len(words)
        unique_words = len(word_counts)
        redundancy_penalty = 1 - (unique_words / total_words) if total_words > 0 else 0
        
        # Get LLM self-evaluation
        rating = self.groq_client.get_rating(modified_prompt)
        
        # Calculate final reward
        weights = self.config.CLARITY_WEIGHTS
        clarity_reward = (weights['lambda1'] * avg_similarity - 
                         weights['lambda2'] * redundancy_penalty + 
                         weights['lambda3'] * rating)
        
        return clarity_reward
    
    def calculate_relevance_reward(self, user_query: str, response: str, user_feedback: Optional[int] = None) -> float:
        """R2 = α·user_rating + β·sentiment_score"""
        
        # Use explicit feedback if available, otherwise neutral
        user_rating = user_feedback if user_feedback is not None else 0
        
        # Calculate sentiment score
        sentiment_scores = self.sentiment_analyzer.polarity_scores(response)
        sentiment_score = sentiment_scores['compound']  # Range: [-1, 1]
        
        # Calculate final reward
        weights = self.config.RELEVANCE_WEIGHTS
        relevance_reward = weights['alpha'] * user_rating + weights['beta'] * sentiment_score
        
        return relevance_reward
    
    def calculate_hallucination_penalty(self, response: str) -> float:
        """R3 = -γ·hallucination_score"""
        
        hallucination_score = 0
        num_checks = 0
        
        # Check with Wikipedia for factual verification
        wiki_score = self._verify_with_wikipedia(response)
        if wiki_score is not None:
            hallucination_score += wiki_score
            num_checks += 1
        
        # LLM self-verification
        verification_prompt = f"Is this statement factually correct? Answer with just 'Yes' or 'No': '{response}'"
        verification_response = self.groq_client.get_response(verification_prompt)
        llm_score = 0.0 if 'yes' in verification_response.lower() else 1.0
        hallucination_score += llm_score
        num_checks += 1
        
        # Average the scores
        if num_checks > 0:
            hallucination_score /= num_checks
        
        penalty = self.config.HALLUCINATION_WEIGHTS['gamma'] * hallucination_score
        return penalty
    
    def _verify_with_wikipedia(self, response: str) -> Optional[float]:
        """Verify response against Wikipedia"""
        try:
            # Extract key terms from response
            words = response.split()
            if len(words) < 3:
                return None
            
            # Search Wikipedia with first few words
            search_term = ' '.join(words[:5])
            search_results = wikipedia.search(search_term, results=3)
            
            if not search_results:
                return 0.8  # High hallucination score if no results
            
            # Get summary of first result
            try:
                summary = wikipedia.summary(search_results[0], sentences=3)
                
                # Calculate similarity between response and Wikipedia summary
                response_embedding = self.embedding_model.encode([response])
                summary_embedding = self.embedding_model.encode([summary])
                
                similarity = cosine_similarity(response_embedding, summary_embedding)[0][0]
                
                # Convert similarity to hallucination score (higher similarity = lower hallucination)
                hallucination_score = 1.0 - similarity
                return hallucination_score
                
            except wikipedia.exceptions.DisambiguationError:
                return 0.3  # Medium hallucination score for ambiguous topics
            except wikipedia.exceptions.PageError:
                return 0.8  # High hallucination score if page doesn't exist
                
        except Exception as e:
            logger.warning(f"Wikipedia verification failed: {e}")
            return None
    
    def calculate_total_reward(self, original_prompt: str, modified_prompt: str, 
                             response: str, user_feedback: Optional[int] = None) -> float:
        """Calculate total reward combining all three components"""
        
        clarity_reward = self.calculate_clarity_reward(original_prompt, modified_prompt)
        relevance_reward = self.calculate_relevance_reward(original_prompt, response, user_feedback)
        hallucination_penalty = self.calculate_hallucination_penalty(response)
        
        total_reward = clarity_reward + relevance_reward - hallucination_penalty
        
        logger.debug(f"Reward breakdown - Clarity: {clarity_reward:.3f}, "
                    f"Relevance: {relevance_reward:.3f}, "
                    f"Hallucination Penalty: {hallucination_penalty:.3f}, "
                    f"Total: {total_reward:.3f}")
        
        return total_reward