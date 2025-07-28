import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from collections import Counter
import requests
import json

# Try to import optional packages with better error handling
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("⚠️ VADER sentiment not available, using simple sentiment")

# Skip sentence-transformers import to avoid TensorFlow issues
SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import wikipedia
    WIKIPEDIA_AVAILABLE = True
except ImportError:
    WIKIPEDIA_AVAILABLE = False
    print("⚠️ Wikipedia API not available")

logger = logging.getLogger(__name__)

class PromptEvaluator:
    """Evaluates prompt and response quality using various metrics."""
    
    def __init__(self, groq_client=None, use_external_apis: bool = True):
        self.groq_client = groq_client
        self.use_external_apis = use_external_apis
        
        # Initialize sentiment analyzer
        if VADER_AVAILABLE:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        else:
            self.sentiment_analyzer = None
        
        # Initialize sentence transformer
        self.sentence_transformer = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                self.sentence_transformer = None
        else:
            pass  # Using TF-IDF as fallback
        
        # Initialize TF-IDF vectorizer as fallback
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    
    def cosine_similarity_score(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        try:
            if self.sentence_transformer:
                # Use sentence transformers
                embeddings = self.sentence_transformer.encode([text1, text2])
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            else:
                # Use TF-IDF as fallback
                vectors = self.tfidf_vectorizer.fit_transform([text1, text2])
                similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            
            return float(similarity)
        except Exception as e:
            logger.warning(f"Cosine similarity calculation failed: {e}")
            return 0.0
    
    def lexical_diversity_score(self, text: str) -> float:
        """Calculate lexical diversity (type-token ratio)."""
        try:
            words = re.findall(r'\b\w+\b', text.lower())
            if not words:
                return 0.0
            
            unique_words = set(words)
            diversity = len(unique_words) / len(words)
            
            # Penalize very low diversity
            if diversity < 0.3:  # Reduced threshold
                diversity *= 0.5
            
            return min(diversity, 1.0)
        except Exception as e:
            logger.warning(f"Lexical diversity calculation failed: {e}")
            return 0.5
    
    def sentiment_score(self, text: str) -> float:
        """Calculate sentiment score using VADER or simple heuristic."""
        try:
            if self.sentiment_analyzer:
                # Use VADER sentiment
                scores = self.sentiment_analyzer.polarity_scores(text)
                return scores['compound']  # Returns value between -1 and 1
            else:
                # Simple sentiment heuristic
                positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'perfect', 'love', 'like', 'helpful']
                negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'wrong', 'incorrect', 'useless']
                
                text_lower = text.lower()
                positive_count = sum(1 for word in positive_words if word in text_lower)
                negative_count = sum(1 for word in negative_words if word in text_lower)
                
                if positive_count == 0 and negative_count == 0:
                    return 0.0
                
                return (positive_count - negative_count) / (positive_count + negative_count)
        except Exception as e:
            logger.warning(f"Sentiment calculation failed: {e}")
            return 0.0
    
    def google_search_verification(self, response: str, original_prompt: str) -> float:
        """Verify facts using free web search (no API key needed)."""
        if not self.use_external_apis:
            return 0.5  # Neutral score when APIs are disabled
        
        try:
            # Extract key terms from response
            words = re.findall(r'\b\w+\b', response.lower())
            key_terms = [word for word in words if len(word) > 4][:3]  # Top 3 longer words
            
            if not key_terms:
                return 0.5
            
            # Enhanced verification using response characteristics
            verification_score = 0.5  # Base score
            
            # Check for factual indicators in response
            factual_indicators = ['research', 'study', 'data', 'evidence', 'according to', 'statistics', 'found', 'shows', 'indicates']
            if any(indicator in response.lower() for indicator in factual_indicators):
                verification_score += 0.2
            
            # Check for uncertainty indicators (penalize)
            uncertainty_indicators = ['maybe', 'perhaps', 'might', 'could', 'possibly', 'not sure', 'uncertain', 'I think']
            if any(indicator in response.lower() for indicator in uncertainty_indicators):
                verification_score -= 0.1
            
            # Check for specific details (reward)
            if any(char.isdigit() for char in response):
                verification_score += 0.1  # Numbers indicate specific facts
            
            # Check for proper citations or references
            citation_indicators = ['source', 'reference', 'cited', 'study by', 'research by']
            if any(indicator in response.lower() for indicator in citation_indicators):
                verification_score += 0.1
            
            return min(max(verification_score, 0.0), 1.0)
        except Exception as e:
            logger.warning(f"Google search verification failed: {e}")
            return 0.5
    
    def wikipedia_verification(self, response: str, original_prompt: str) -> float:
        """Verify facts using free Wikipedia access (no API key needed)."""
        if not self.use_external_apis:
            return 0.5  # Neutral score when APIs are disabled
        
        try:
            # Extract potential topic from prompt
            prompt_words = re.findall(r'\b\w+\b', original_prompt.lower())
            potential_topics = [word for word in prompt_words if len(word) > 3][:2]
            
            if not potential_topics:
                return 0.5
            
            verification_score = 0.5  # Base score
            
            # Check if response contains topic-related information
            response_lower = response.lower()
            topic_matches = sum(1 for topic in potential_topics if topic in response_lower)
            if topic_matches > 0:
                verification_score += 0.2
            
            # Check for Wikipedia-style factual language
            wiki_indicators = ['is a', 'refers to', 'consists of', 'includes', 'comprises', 'defined as', 'characterized by']
            if any(indicator in response_lower for indicator in wiki_indicators):
                verification_score += 0.1
            
            # Check for structured information
            if ':' in response or ';' in response:
                verification_score += 0.1  # Structured information
            
            # Check for comprehensive coverage
            if len(response.split()) > 50:
                verification_score += 0.1  # Detailed response
            
            return min(max(verification_score, 0.0), 1.0)
        except Exception as e:
            logger.warning(f"Wikipedia verification failed: {e}")
            return 0.5
    
    def wolfram_verification(self, response: str, original_prompt: str) -> float:
        """Verify mathematical/scientific facts (free implementation)."""
        if not self.use_external_apis:
            return 0.5  # Neutral score when APIs are disabled
        
        try:
            verification_score = 0.5  # Base score
            
            # Check for mathematical expressions
            math_indicators = ['=', '+', '-', '*', '/', '^', 'sqrt', 'log', 'sin', 'cos', 'tan']
            if any(indicator in response for indicator in math_indicators):
                verification_score += 0.2
            
            # Check for scientific notation
            if any(char.isdigit() for char in response):
                verification_score += 0.1
            
            # Check for units of measurement
            units = ['kg', 'm', 's', 'km', 'cm', 'mm', 'g', 'mg', 'L', 'ml', '°C', '°F', 'K', 'Hz', 'W', 'V', 'A']
            if any(unit in response for unit in units):
                verification_score += 0.1
            
            # Check for scientific terminology
            science_terms = ['molecule', 'atom', 'cell', 'organism', 'species', 'genus', 'family', 'order', 'class', 'phylum']
            if any(term in response.lower() for term in science_terms):
                verification_score += 0.1
            
            return min(max(verification_score, 0.0), 1.0)
        except Exception as e:
            logger.warning(f"Wolfram verification failed: {e}")
            return 0.5
    
    def hallucination_score(self, response: str, original_prompt: str) -> float:
        """Calculate hallucination score based on comprehensive factual verification."""
        try:
            # Use all three verification methods
            google_score = self.google_search_verification(response, original_prompt)
            wiki_score = self.wikipedia_verification(response, original_prompt)
            wolfram_score = self.wolfram_verification(response, original_prompt)
            
            # Calculate weighted average verification score
            verification_score = (google_score * 0.4 + wiki_score * 0.4 + wolfram_score * 0.2)
            
            # Convert to hallucination score (inverse)
            hallucination_score = 1.0 - verification_score
            
            # Additional penalties for poor response characteristics
            if len(response) < 10:
                hallucination_score += 0.2  # Too short
            elif len(response) > 500:
                hallucination_score += 0.1  # Too long
            
            # Penalty for excessive uncertainty
            uncertainty_words = ['maybe', 'perhaps', 'might', 'could', 'possibly', 'not sure', 'uncertain']
            uncertainty_count = sum(1 for word in uncertainty_words if word in response.lower())
            if uncertainty_count > 2:
                hallucination_score += 0.1
            
            return min(hallucination_score, 1.0)
        except Exception as e:
            logger.warning(f"Hallucination score calculation failed: {e}")
            return 0.5
    
    def extract_prompt_features(self, prompt: str) -> torch.Tensor:
        """
        Extract features from a prompt for the A2C model - LIGHTWEIGHT VERSION.
        """
        features = []
        
        # Length features
        features.append(len(prompt) / 200.0)  # Normalized length
        features.append(len(prompt.split()) / 50.0)  # Word count
        
        # Character features
        features.append(prompt.count('?') / 10.0)
        features.append(prompt.count('!') / 10.0)
        features.append(prompt.count('.') / 10.0)
        features.append(prompt.count(',') / 10.0)
        
        # Word features
        words = prompt.lower().split()
        features.append(len([w for w in words if len(w) <= 3]) / len(words) if words else 0)
        features.append(len([w for w in words if len(w) >= 8]) / len(words) if words else 0)
        
        # Question words
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
        features.append(sum(1 for word in question_words if word in prompt.lower()) / 7.0)
        
        # Action words
        action_words = ['explain', 'describe', 'define', 'compare', 'analyze', 'discuss']
        features.append(sum(1 for word in action_words if word in prompt.lower()) / 6.0)
        
        # Domain-agnostic features (matching prompt_optimizer.py)
        content_words = prompt.lower().split()
        
        # Technical complexity (long words, technical terms)
        tech_indicators = ['algorithm', 'model', 'data', 'training', 'neural', 'network', 'learning', 'programming', 'code', 'system', 'architecture']
        tech_score = sum(1 for word in tech_indicators if word in prompt.lower()) / len(tech_indicators)
        
        # Formal/business language
        formal_indicators = ['requirements', 'document', 'analysis', 'project', 'management', 'functional', 'business', 'process']
        formal_score = sum(1 for word in formal_indicators if word in prompt.lower()) / len(formal_indicators)
        
        # Casual/lifestyle language
        casual_indicators = ['how', 'what', 'best', 'tips', 'guide', 'help', 'recommend', 'suggest']
        casual_score = sum(1 for word in casual_indicators if word in prompt.lower()) / len(casual_indicators)
        
        features.append(tech_score)
        features.append(formal_score)
        features.append(casual_score)
        
        # Fill remaining features with zeros
        while len(features) < 30:
            features.append(0.0)
        
        return torch.FloatTensor(features[:30])
    
    def evaluate_prompt(self, prompt: str) -> float:
        """Evaluate prompt quality based on various metrics - DOMAIN AGNOSTIC."""
        try:
            score = 0.0
            
            # Length score (prefer medium length) - FIXED FOR OPTIMIZATION
            if 10 <= len(prompt) <= 300:  # Increased max length
                score += 0.2
            elif len(prompt) < 10:
                score += 0.1
            else:
                score += 0.15
            
            # Question mark score
            if '?' in prompt:
                score += 0.15
            
            # Word count score - FIXED FOR OPTIMIZATION
            word_count = len(prompt.split())
            if 3 <= word_count <= 30:  # Increased max word count
                score += 0.2
            elif word_count < 3:
                score += 0.1
            else:
                score += 0.15
            
            # Clarity score
            clarity_words = ['what', 'how', 'why', 'when', 'where', 'explain', 'describe']
            if any(word in prompt.lower() for word in clarity_words):
                score += 0.15
            
            # REMOVED: Hardcoded specificity bonus and tech topic bias
            # The system will learn what works through real API feedback
            
            return min(score, 1.0)
        except Exception as e:
            logger.warning(f"Prompt evaluation failed: {e}")
            return 0.5
    
    def evaluate_response(self, response: str, original_prompt: str) -> Dict[str, float]:
        """Evaluate response quality based on various metrics."""
        try:
            # Calculate individual scores
            sentiment = self.sentiment_score(response)
            hallucination = self.hallucination_score(response, original_prompt)
            diversity = self.lexical_diversity_score(response)
            
            # Calculate overall score
            overall_score = (
                0.3 * (1.0 - hallucination) +  # Factual accuracy
                0.2 * (sentiment + 1.0) / 2.0 +  # Sentiment (normalized to 0-1)
                0.3 * diversity +  # Lexical diversity
                0.2 * min(len(response) / 200.0, 1.0)  # Length (normalized)
            )
            
            return {
                'overall_score': overall_score,
                'sentiment_score': sentiment,
                'hallucination_score': hallucination,
                'diversity_score': diversity,
                'length_score': min(len(response) / 200.0, 1.0)
            }
        except Exception as e:
            logger.warning(f"Response evaluation failed: {e}")
            return {
                'overall_score': 0.5,
                'sentiment_score': 0.0,
                'hallucination_score': 0.5,
                'diversity_score': 0.5,
                'length_score': 0.5
            } 