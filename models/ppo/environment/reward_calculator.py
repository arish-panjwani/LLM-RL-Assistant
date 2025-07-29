import requests
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
import wikipedia
import logging
from utils.groq_client import GroqClient
from utils.optimization_strategies import OptimizationStrategies
from typing import Optional
from config.config import Config

logger = logging.getLogger(__name__)

class RewardCalculator:
    def __init__(self, groq_client: GroqClient, config: Config):
        self.groq_client = groq_client
        self.config = config
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        nltk.download('punkt', quiet=True)
    
    def _calculate_clarity_score(self, prompt: str) -> float:
        """Calculate clarity score based on prompt structure and complexity"""
        try:
            # Check sentence structure
            sentences = sent_tokenize(prompt)
            if not sentences:
                return 0.0
                
            # Calculate average sentence length (penalize very long sentences)
            avg_length = np.mean([len(s.split()) for s in sentences])
            length_score = max(0, 1 - (abs(avg_length - 15) / 30))
            
            # Check for question clarity
            has_question_word = any(word in prompt.lower() for word in ['what', 'why', 'how', 'when', 'where', 'who'])
            question_score = 0.8 if has_question_word else 0.5
            
            # Check sentiment neutrality (prefer neutral/objective prompts)
            sentiment_scores = self.sentiment_analyzer.polarity_scores(prompt)
            neutrality_score = 1 - abs(sentiment_scores['compound'])
            
            # Combine scores with weights
            final_score = (
                0.4 * length_score +
                0.3 * question_score +
                0.3 * neutrality_score
            )
            
            return max(0, min(1, final_score))
            
        except Exception as e:
            print(f"Error calculating clarity score: {e}")
            return 0.5
    
    def calculate_total_reward(self, original_prompt: str, modified_prompt: str, response: str, **kwargs) -> float:
        """Calculate total reward with optional context parameters"""
        clarity = self._calculate_clarity_score(modified_prompt)
        relevance = self._calculate_relevance_score(original_prompt, modified_prompt)
        hallucination = self._calculate_hallucination_score(response)
        
        # Apply context-specific adjustments
        context = kwargs.get('context', 'general')
        depth = kwargs.get('depth', 'intermediate')
        
        if context == 'technical' or depth == 'expert':
            clarity *= 1.2
        elif context == 'practical' or depth == 'beginner':
            clarity *= 0.8
            
        return (clarity + relevance - hallucination) / 3.0
    
    def _calculate_relevance_score(self, original: str, modified: str) -> float:
        """Calculate semantic similarity between original and modified prompts"""
        try:
            embeddings = self.embedding_model.encode([original, modified])
            similarity = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
            return max(0, min(1, similarity))
        except Exception as e:
            print(f"Error calculating relevance score: {e}")
            return 0.5
    
    def _calculate_hallucination_score(self, response: str) -> float:
        """Calculate likelihood of hallucination in response"""
        try:
            # Check response length (very short responses might be low quality)
            if len(response.split()) < 10:
                return 0.8
                
            # Check for uncertainty markers
            uncertainty_phrases = ['might be', 'could be', 'possibly', 'perhaps', 'maybe']
            uncertainty_score = sum(1 for phrase in uncertainty_phrases if phrase in response.lower())
            uncertainty_penalty = min(0.5, uncertainty_score * 0.1)
            
            # Default to moderate hallucination score if no clear indicators
            return max(0, min(1, 0.3 + uncertainty_penalty))
            
        except Exception as e:
            print(f"Error calculating hallucination score: {e}")
            return 0.5
    
    def calculate_metrics(self, original_prompt: str, optimized_prompt: str, response: str, context: str = "general", depth: str = "intermediate") -> dict:
        """Calculate comprehensive metrics for prompt optimization"""
        try:
            # Calculate individual scores
            clarity_score = float(self._calculate_clarity_score(optimized_prompt))
            relevance_score = float(self._calculate_relevance_score(original_prompt, optimized_prompt))
            hallucination_penalty = float(self._calculate_hallucination_score(response))
            
            # Calculate diversity score based on prompt variation
            diversity_score = float(self._calculate_diversity_score(original_prompt, optimized_prompt))
            
            # Apply context-specific adjustments
            if context == 'technical' or depth == 'expert':
                clarity_score *= 1.2
                relevance_score *= 1.1
            elif context == 'practical' or depth == 'beginner':
                clarity_score *= 0.9
                relevance_score *= 0.95
            
            # Ensure scores are within bounds
            clarity_score = max(0, min(1, clarity_score))
            relevance_score = max(0, min(1, relevance_score))
            hallucination_penalty = max(0, min(1, hallucination_penalty))
            diversity_score = max(0, min(1, diversity_score))
            
            return {
                "clarity_score": round(float(clarity_score), 3),
                "relevance_score": round(float(relevance_score), 3),
                "hallucination_penalty": round(float(hallucination_penalty), 3),
                "diversity_score": round(float(diversity_score), 3)
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {
                "clarity_score": 0.5,
                "relevance_score": 0.5,
                "hallucination_penalty": 0.3,
                "diversity_score": 0.5
            }
    
    def _calculate_diversity_score(self, original: str, optimized: str) -> float:
        """Calculate diversity score based on prompt variation"""
        try:
            # Calculate embedding similarity
            embeddings = self.embedding_model.encode([original, optimized])
            similarity = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
            
            # Diversity is inverse of similarity (but not too different)
            diversity = 1 - similarity
            
            # Penalize if too similar or too different
            if diversity < 0.1:  # Too similar
                diversity = 0.1
            elif diversity > 0.8:  # Too different
                diversity = 0.8
                
            return float(diversity)
            
        except Exception as e:
            logger.error(f"Error calculating diversity score: {e}")
            return 0.5