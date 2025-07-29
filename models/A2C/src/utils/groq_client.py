import requests
import json
import logging
from typing import Dict, Any, Optional, List
import time
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class GroqClient:
    """Client for interacting with Groq API."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.groq.com/openai/v1"):
        """
        Initialize Groq client.
        
        Args:
            api_key: Groq API key (if None, will try to get from environment)
            base_url: Base URL for Groq API
        """
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("Groq API key not provided. Set GROQ_API_KEY environment variable.")
        
        self.base_url = base_url
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        # Available models
        self.models = {
            'llama3-8b': 'llama3-8b-8192',
            'llama3-70b': 'llama3-70b-8192',
            'mixtral': 'mixtral-8x7b-32768',
            'gemma': 'gemma-7b-it'
        }
        
        logger.info("Groq client initialized successfully")
    
    def generate_response(self, 
                         prompt: str, 
                         model: str = 'llama3-8b',
                         max_tokens: int = 1000,
                         temperature: float = 0.7,
                         top_p: float = 1.0,
                         frequency_penalty: float = 0.0,
                         presence_penalty: float = 0.0) -> Dict[str, Any]:
        """
        Generate response from Groq API.
        
        Args:
            prompt: Input prompt
            model: Model to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
            
        Returns:
            API response dictionary
        """
        if model not in self.models:
            raise ValueError(f"Model {model} not supported. Available models: {list(self.models.keys())}")
        
        payload = {
            'model': self.models[model],
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'frequency_penalty': frequency_penalty,
            'presence_penalty': presence_penalty
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return {
                'success': True,
                'response': result['choices'][0]['message']['content'],
                'usage': result['usage'],
                'model': model
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': None
            }
    
    def evaluate_prompt_quality(self, prompt: str) -> Dict[str, Any]:
        """
        Use Groq to self-evaluate prompt quality.
        
        Args:
            prompt: Prompt to evaluate
            
        Returns:
            Evaluation results
        """
        evaluation_prompt = f"""
        Rate the clarity and specificity of this prompt on a scale of 1-10.
        Consider:
        1. Clarity: Is the prompt clear and easy to understand?
        2. Specificity: Does it provide enough detail for a good response?
        3. Completeness: Does it include all necessary information?
        
        Prompt: "{prompt}"
        
        Please provide:
        1. A numerical score (1-10)
        2. Brief explanation of your rating
        3. Suggestions for improvement
        
        Format your response as JSON:
        {{
            "score": <number>,
            "explanation": "<text>",
            "suggestions": ["<suggestion1>", "<suggestion2>"]
        }}
        """
        
        result = self.generate_response(
            evaluation_prompt,
            temperature=0.3,
            max_tokens=500
        )
        
        if result['success']:
            try:
                # Try to parse JSON response
                import re
                json_match = re.search(r'\{.*\}', result['response'], re.DOTALL)
                if json_match:
                    evaluation = json.loads(json_match.group())
                    return {
                        'success': True,
                        'score': evaluation.get('score', 5),
                        'explanation': evaluation.get('explanation', ''),
                        'suggestions': evaluation.get('suggestions', [])
                    }
                else:
                    # Fallback: extract score from text
                    import re
                    score_match = re.search(r'score["\s:]*(\d+)', result['response'], re.IGNORECASE)
                    score = int(score_match.group(1)) if score_match else 5
                    return {
                        'success': True,
                        'score': score,
                        'explanation': result['response'],
                        'suggestions': []
                    }
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse evaluation response: {e}")
                return {
                    'success': False,
                    'error': f"Failed to parse response: {e}",
                    'score': 5
                }
        else:
            return {
                'success': False,
                'error': result['error'],
                'score': 5
            }
    
    def verify_facts(self, statement: str) -> Dict[str, Any]:
        """
        Use Groq to verify factual accuracy of a statement.
        
        Args:
            statement: Statement to verify
            
        Returns:
            Verification results
        """
        verification_prompt = f"""
        Verify if this statement is factually correct. Consider:
        1. Is this statement accurate based on widely accepted knowledge?
        2. Are there any factual errors or inconsistencies?
        3. What is your confidence level in this verification?
        
        Statement: "{statement}"
        
        Please provide:
        1. A confidence score (0-100%)
        2. Whether the statement is factually correct (true/false/uncertain)
        3. Brief explanation
        4. Any corrections if needed
        
        Format your response as JSON:
        {{
            "confidence": <number>,
            "is_correct": "<true/false/uncertain>",
            "explanation": "<text>",
            "corrections": ["<correction1>", "<correction2>"]
        }}
        """
        
        result = self.generate_response(
            verification_prompt,
            temperature=0.2,
            max_tokens=500
        )
        
        if result['success']:
            try:
                import re
                json_match = re.search(r'\{.*\}', result['response'], re.DOTALL)
                if json_match:
                    verification = json.loads(json_match.group())
                    return {
                        'success': True,
                        'confidence': verification.get('confidence', 50),
                        'is_correct': verification.get('is_correct', 'uncertain'),
                        'explanation': verification.get('explanation', ''),
                        'corrections': verification.get('corrections', [])
                    }
                else:
                    return {
                        'success': True,
                        'confidence': 50,
                        'is_correct': 'uncertain',
                        'explanation': result['response'],
                        'corrections': []
                    }
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse verification response: {e}")
                return {
                    'success': False,
                    'error': f"Failed to parse response: {e}",
                    'confidence': 50,
                    'is_correct': 'uncertain'
                }
        else:
            return {
                'success': False,
                'error': result['error'],
                'confidence': 50,
                'is_correct': 'uncertain'
            }
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return list(self.models.keys())
    
    def test_connection(self) -> bool:
        """Test API connection."""
        try:
            result = self.generate_response("Hello", max_tokens=10)
            return result['success']
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False 