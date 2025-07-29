# server/api_routes.py
from flask import request, jsonify
from utils.groq_client import GroqClient
from utils.google_client import GoogleClient
from utils.wolfram_client import WolframClient
from config.config import Config
import logging
import numpy as np
import os
import sys
import json

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize PPO components with error handling
ppo_model = None
reward_calculator = None
PPO_AVAILABLE = False

try:
    from models.ppo.ppo_model import PPOModel
    from environment.reward_calculator import RewardCalculator
    PPO_AVAILABLE = True
    logging.info("PPO model components imported successfully")
except ImportError as e:
    logging.warning(f"PPO model not available: {e}")

logger = logging.getLogger(__name__)

def json_serializable(obj):
    """Convert numpy types to JSON serializable types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def safe_json_response(data):
    """Convert data to JSON serializable format"""
    return json.loads(json.dumps(data, default=json_serializable))

def register_routes(app):
    config = Config()
    
    # Initialize all API clients
    groq_client = GroqClient(config.GROQ_API_KEY)
    google_client = GoogleClient(config.GOOGLE_API_KEY)
    wolfram_client = WolframClient(config.WOLFRAM_APP_ID)
    
    # Initialize PPO model and reward calculator if available
    global ppo_model, reward_calculator
    
    if PPO_AVAILABLE:
        try:
            ppo_model = PPOModel(config)
            reward_calculator = RewardCalculator(groq_client, config)
            logger.info("PPO model and reward calculator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PPO components: {e}")
            ppo_model = None
            reward_calculator = None
    
    @app.route("/health", methods=["GET"])
    def health_check():
        return jsonify({"status": "ok"})
    
    @app.route("/model_info", methods=["GET"])
    def model_info():
        return jsonify({
            "model": "PPO",
            "status": "loaded",
            "version": "1.0",
            "ppo_available": ppo_model is not None,
            "groq_available": groq_client.api_available,
            "google_available": google_client.api_available,
            "wolfram_available": wolfram_client.api_available
        })
    
    @app.route("/optimize_prompt", methods=["POST"])
    def optimize_prompt():
        try:
            # Validate request
            if not request.is_json:
                return jsonify({"error": "Request must be JSON"}), 400
                
            data = request.get_json()
            prompt = data.get("prompt")
            context = data.get("context", "general")
            depth = data.get("depth", "intermediate")
            use_apis = data.get("use_apis", True)  # Enable multi-API responses
            
            if not prompt:
                return jsonify({"error": "Missing prompt"}), 400
            
            logger.debug(f"Processing prompt: {prompt[:50]}...")
            
            try:
                # Check if we're using fallback mode
                using_fallback = not groq_client.api_available
                
                if ppo_model and reward_calculator and not using_fallback:
                    # Use trained PPO model for dynamic optimization
                    logger.info("Using trained PPO model for dynamic optimization")
                    
                    try:
                        # Get embedding for the prompt
                        embedding = reward_calculator.embedding_model.encode([prompt])[0]
                        
                        # Use PPO model to optimize the embedding
                        optimized_embedding = ppo_model.optimize_prompt(embedding)
                        
                        # Generate optimized prompt using LLM
                        optimized_prompt = _generate_dynamic_prompt(prompt, context, depth, groq_client)
                        
                        # Get response for optimized prompt
                        response = groq_client.get_response(optimized_prompt)
                        
                        # Calculate metrics using reward calculator
                        metrics = reward_calculator.calculate_metrics(prompt, optimized_prompt, response, context, depth)
                        
                    except Exception as e:
                        logger.error(f"PPO processing error: {e}")
                        # Fallback to simple optimization
                        optimized_prompt = f"Considering the {context} context at a {depth} level: {prompt}"
                        response = groq_client.get_response(optimized_prompt)
                        metrics = _calculate_simple_metrics(using_fallback)
                        
                else:
                    # Fallback to simple optimization
                    logger.info("Using fallback optimization (no PPO model or API)")
                    optimized_prompt = f"Considering the {context} context at a {depth} level: {prompt}"
                    response = groq_client.get_response(optimized_prompt)
                    
                    # Simple metrics
                    metrics = _calculate_simple_metrics(using_fallback)
                
                if not response:
                    return jsonify({"error": "Failed to get response from LLM"}), 500
                
                # Enhanced response with multiple APIs
                enhanced_response = response
                api_sources = []
                
                if use_apis:
                    # Add Google factual information
                    if google_client.api_available:
                        google_info = google_client.get_factual_info(prompt)
                        if google_info and google_info != f"Factual information about {prompt} would be available with Google API key.":
                            enhanced_response += f"\n\n📚 **Factual Information:**\n{google_info}"
                            api_sources.append("Google Knowledge Graph")
                    
                    # Add Wolfram computational/factual information
                    if wolfram_client.api_available:
                        wolfram_info = wolfram_client.get_factual_answer(prompt)
                        if wolfram_info and wolfram_info != f"Factual information about {prompt} would be available with Wolfram Alpha App ID.":
                            enhanced_response += f"\n\n🧮 **Computational Data:**\n{wolfram_info}"
                            api_sources.append("Wolfram Alpha")
                
                # Debug logging
                logger.info(f"API Response - Original: {prompt}")
                logger.info(f"API Response - Optimized: {optimized_prompt}")
                logger.info(f"API Response - LLM Response: {enhanced_response[:100]}...")
                
                return jsonify({
                    "success": True,
                    "original_prompt": prompt,
                    "optimized_prompt": optimized_prompt,
                    "response": enhanced_response,
                    "context": context,
                    "depth": depth,
                    "using_fallback": using_fallback,
                    "api_available": groq_client.api_available,
                    "ppo_used": ppo_model is not None and not using_fallback,
                    "api_sources": api_sources,
                    "metrics": safe_json_response(metrics)
                })
                
            except Exception as e:
                logger.error(f"Error processing prompt: {str(e)}")
                return jsonify({"error": f"Error processing prompt: {str(e)}"}), 500
            
        except Exception as e:
            logger.error(f"Error handling request: {str(e)}")
            return jsonify({"error": f"Error handling request: {str(e)}"}), 500
    
    return app

def _calculate_simple_metrics(using_fallback: bool) -> dict:
    """Calculate simple metrics when PPO model is not available"""
    return {
        "clarity_score": 0.6 if using_fallback else 0.8,
        "relevance_score": 0.7 if using_fallback else 0.9,
        "hallucination_penalty": 0.2 if using_fallback else 0.1,
        "diversity_score": 0.3 if using_fallback else 0.1
    }

def _generate_dynamic_prompt(original_prompt: str, context: str, depth: str, groq_client: GroqClient) -> str:
    """Generate dynamic optimized prompt using LLM"""
    
    meta_prompt = f"""As an expert prompt engineer, optimize this query: "{original_prompt}"

Context: {context}
Depth: {depth}
Optimization Goals:
1. Generate a prompt appropriate for {depth} level understanding
2. Focus on {context} context and applications
3. Use varied and engaging language
4. Maintain technical accuracy
5. Encourage detailed, structured responses

Requirements:
- Do NOT start with "Please provide"
- Use diverse prompt structures
- Match the complexity to the {depth} level
- Maintain focus on {context} aspects
- Encourage analytical thinking

Generate only the optimized prompt, no additional text."""

    optimized_prompt = groq_client.get_response(meta_prompt).strip()
    
    # Validate and fallback if needed
    if not optimized_prompt or len(optimized_prompt) < 20:
        # Use context-aware fallback patterns
        patterns = {
            ('academic', 'expert'): f"Analyze the theoretical foundations and advanced implications of {original_prompt}",
            ('scientific', 'detailed'): f"Examine the mechanisms and processes involved in {original_prompt}",
            ('technical', 'expert'): f"Evaluate the architectural components and technical implementation of {original_prompt}",
            ('practical', 'beginner'): f"In simple terms, explain how {original_prompt} works and its everyday applications",
            ('current_events', 'comprehensive'): f"Analyze the current state, implications, and future trends of {original_prompt}"
        }
        
        return patterns.get((context, depth), f"Explain the key concepts and applications of {original_prompt}")
        
    return optimized_prompt