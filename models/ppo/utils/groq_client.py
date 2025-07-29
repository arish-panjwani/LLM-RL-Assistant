import groq
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class GroqClient:
    """Client for interacting with Groq API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = "llama3-8b-8192"  # Updated to a more reliable model
        self.fallback_model = "llama3-70b-8192"  # Fallback model
        
        # Check if API key is provided
        if not api_key or api_key.strip() == '':
            logger.warning("No Groq API key provided. Using fallback responses only.")
            self.client = None
            self.api_available = False
        else:
            try:
                self.client = groq.Client(api_key=api_key)
                self.api_available = True
                logger.info("Groq client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")
                self.client = None
                self.api_available = False
        
    def get_response(self, prompt: str) -> str:
        """Get response from Groq API with fallback and error handling"""
        if not prompt or not isinstance(prompt, str):
            logger.error("Invalid prompt provided")
            return ""
        
        # If no API key or client not available, use fallback
        if not self.api_available or not self.client:
            logger.info("Using fallback response due to missing API key")
            return self._get_fallback_response(prompt)
            
        try:
            completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert prompt engineer focused on creating clear, specific, and contextually appropriate prompts."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                temperature=0.7,
                max_tokens=1024
            )
            return completion.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            # Try fallback model if primary fails
            try:
                logger.warning(f"Primary model failed, trying fallback model: {e}")
                completion = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert prompt engineer focused on creating clear, specific, and contextually appropriate prompts."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    model=self.fallback_model,
                    temperature=0.7,
                    max_tokens=1024
                )
                return completion.choices[0].message.content
            except Exception as fallback_error:
                logger.error(f"Fallback model also failed: {fallback_error}")
                return self._get_fallback_response(prompt)
            
    def _get_fallback_response(self, prompt: str) -> str:
        """Generate fallback response when API fails"""
        context = "general"
        depth = "intermediate"
        
        # Enhanced fallback patterns based on prompt content
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ["machine learning", "ai", "artificial intelligence"]):
            return f"Explain the fundamental concepts and applications of {prompt} in a clear and accessible manner."
        elif any(word in prompt_lower for word in ["quantum", "physics", "computing"]):
            return f"Describe the key principles and practical implications of {prompt} for both experts and beginners."
        elif any(word in prompt_lower for word in ["photosynthesis", "biology", "science"]):
            return f"Explain the biological processes and mechanisms involved in {prompt} with relevant examples."
        elif any(word in prompt_lower for word in ["climate", "environment", "change"]):
            return f"Analyze the current understanding and future implications of {prompt} from multiple perspectives."
        elif any(word in prompt_lower for word in ["food", "restaurant", "cuisine", "dish", "meal", "eat", "dining"]):
            return f"Based on culinary knowledge and cultural context, here's information about {prompt}: This appears to be a traditional dish with cultural significance. For authentic experiences, consider local restaurants, cultural centers, or community recommendations that specialize in traditional cuisine."
        else:
            return f"Provide a comprehensive explanation of {prompt} covering its key aspects, applications, and significance."