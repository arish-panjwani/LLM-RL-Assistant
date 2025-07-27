import groq
from typing import Optional, Dict, Any
import logging
import time
import random
import hashlib
import json

logger = logging.getLogger(__name__)

class GroqClient:
    def __init__(self, api_key: str):
        if not api_key:
            logger.warning("No Groq API key provided, using mock responses")
            self.client = None
        else:
            self.client = groq.Groq(api_key=api_key)
        
        # Rate limiting and caching
        self.last_call_time = 0
        self.call_delay = 1.0  # 1 second between calls
        self.response_cache = {}
        
        # Mock responses for testing without API key
        self.mock_responses = [
            "This is a helpful response to your query.",
            "I understand your question and here's my answer.",
            "Thank you for asking. Here's what I think.",
            "That's an interesting question. Let me explain.",
            "I can help you with that. Here's the information you need."
        ]
    
    def _rate_limit(self):
        """Ensure we don't exceed rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_call_time
        if time_since_last < self.call_delay:
            time.sleep(self.call_delay - time_since_last)
        self.last_call_time = time.time()
    
    def _get_cache_key(self, prompt: str, model: str) -> str:
        """Generate cache key for prompt and model"""
        return hashlib.md5(f"{prompt}:{model}".encode()).hexdigest()
    
    def get_response(self, prompt: str, model: str = "llama3-8b-8192") -> str:
        """Get response from Groq API or return mock response"""
        if self.client is None:
            # Return mock response for testing
            time.sleep(0.1)  # Simulate API delay
            return random.choice(self.mock_responses) + f" (Query: {prompt[:50]}...)"
        
        # Check cache first
        cache_key = self._get_cache_key(prompt, model)
        if cache_key in self.response_cache:
            logger.debug(f"Using cached response for prompt: {prompt[:30]}...")
            return self.response_cache[cache_key]
        
        # Rate limiting
        self._rate_limit()
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7
            )
            result = response.choices[0].message.content
            
            # Cache the response
            self.response_cache[cache_key] = result
            logger.debug(f"API call successful for prompt: {prompt[:30]}...")
            
            return result
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return f"Error: Could not get response. {str(e)}"
    
    def get_rating(self, prompt: str) -> float:
        """Get clarity rating for a prompt"""
        rating_prompt = f"Rate the clarity and specificity of this prompt on a scale of 1-10: '{prompt}'"
        response = self.get_response(rating_prompt)
        
        # Extract rating from response
        try:
            # Look for numbers in the response
            import re
            numbers = re.findall(r'\d+(?:\.\d+)?', response)
            if numbers:
                rating = float(numbers[0])
                return min(max(rating, 1.0), 10.0) / 10.0  # Normalize to [0,1]
        except:
            pass
        
        # Default rating if extraction fails
        return 0.7