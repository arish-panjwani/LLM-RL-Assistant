#!/usr/bin/env python3
"""
Flask web application for A2C Prompt Optimization Demo.
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Setup logging - only show warnings and errors
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# History file path
HISTORY_FILE = Path(__file__).parent / "optimization_history.json"

def load_history():
    """Load optimization history from file."""
    try:
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"Error loading history: {e}")
        return []

def save_history(history):
    """Save optimization history to file."""
    try:
        HISTORY_FILE.parent.mkdir(exist_ok=True)
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"Error saving history: {e}")

def add_to_history(optimization_data):
    """Add a new optimization to history."""
    try:
        history = load_history()
        
        # Create history entry
        history_entry = {
            'id': len(history) + 1,
            'timestamp': datetime.now().isoformat(),
            'original_prompt': optimization_data.get('original_prompt', ''),
            'optimized_prompt': optimization_data.get('optimized_prompt', ''),
            'initial_score': optimization_data.get('initial_score', 0),
            'final_score': optimization_data.get('final_score', 0),
            'improvement': optimization_data.get('improvement', 0),
            'action_name': optimization_data.get('action_name', 'Optimization Applied'),
            'optimization_history': optimization_data.get('optimization_history', [])
        }
        
        # Add to history (keep last 50 entries)
        history.append(history_entry)
        if len(history) > 50:
            history = history[-50:]
        
        save_history(history)
        return history
    except Exception as e:
        print(f"Error adding to history: {e}")
        return []

def initialize_components():
    """Initialize all components for the demo."""
    try:
        # Load configuration
        from utils.config import Config
        config = Config()
        
        # Initialize Groq client with environment variable
        from utils.groq_client import GroqClient
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            print("⚠️  GROQ_API_KEY not found - LLM features will be limited")
            groq_client = None
        else:
            groq_client = GroqClient(api_key=groq_api_key)
        
        # Initialize prompt optimizer
        from models.prompt_optimizer import PromptOptimizer
        model_path = "data/models/a2c_domain_agnostic_best.pth"
        optimizer = PromptOptimizer(model_path, config.get_model_config(), groq_client)
        
        return optimizer, groq_client
        
    except Exception as e:
        print(f"❌ Failed to initialize components: {e}")
        return None, None

# Initialize components at startup
optimizer, groq_client = initialize_components()

@app.route('/')
def index():
    """Main demo page."""
    return render_template('index.html')

@app.route('/optimize', methods=['POST'])
def optimize_prompt():
    """Optimize a prompt using the A2C model."""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '').strip()
        
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        if not optimizer:
            return jsonify({'error': 'Optimizer not initialized'}), 500
        
        # Optimize the prompt
        result = optimizer.optimize_prompt(prompt)
        
        # Get LLM responses for both original and optimized prompts
        original_response = "LLM response will be available when API is configured"
        optimized_response = "LLM response will be available when API is configured"
        
        # Get detailed evaluation metrics
        from utils.evaluation_metrics import PromptEvaluator
        evaluator = PromptEvaluator(groq_client, use_external_apis=True)
        
        # Evaluate original prompt
        original_metrics = evaluator.evaluate_response(original_response, prompt) if original_response != "LLM response will be available when API is configured" else {
            'overall_score': result['initial_score'],
            'sentiment_score': 0.0,
            'hallucination_score': 0.5,
            'diversity_score': 0.5,
            'length_score': 0.5
        }
        
        # Evaluate optimized prompt
        optimized_metrics = evaluator.evaluate_response(optimized_response, prompt) if optimized_response != "LLM response will be available when API is configured" else {
            'overall_score': result['final_score'],
            'sentiment_score': 0.0,
            'hallucination_score': 0.5,
            'diversity_score': 0.5,
            'length_score': 0.5
        }
        
        if groq_client:
            try:
                # Get original prompt response
                original_result = groq_client.generate_response(prompt, max_tokens=100)
                if original_result.get('success'):
                    original_response = original_result.get('response', 'No response received')
                    # Re-evaluate with real response
                    original_metrics = evaluator.evaluate_response(original_response, prompt)
                else:
                    original_response = "API temporarily unavailable"
                
                # Get optimized prompt response
                optimized_result = groq_client.generate_response(result['optimized_prompt'], max_tokens=100)
                if optimized_result.get('success'):
                    optimized_response = optimized_result.get('response', 'No response received')
                    # Re-evaluate with real response
                    optimized_metrics = evaluator.evaluate_response(optimized_response, prompt)
                else:
                    optimized_response = "API temporarily unavailable"
                    
            except Exception as e:
                # Log error but don't show to user
                original_response = "API temporarily unavailable"
                optimized_response = "API temporarily unavailable"
        
        # Get optimization history
        optimization_history = result.get('optimization_history', [])
        actions_applied = [step.get('action_name', 'Unknown') for step in optimization_history]
        
        # Prepare clean response for user
        response_data = {
            'original_prompt': prompt,
            'optimized_prompt': result['optimized_prompt'],
            'initial_score': round(result['initial_score'], 3),
            'final_score': round(result['final_score'], 3),
            'improvement': round(result['total_improvement'], 3),
            'original_response': original_response,
            'optimized_response': optimized_response,
            'action_name': result.get('optimization_history', [{}])[0].get('action_name', 'Optimization Applied'),
            'status': 'success',
            
            # Detailed metrics
            'original_metrics': {
                'sentiment': round(original_metrics.get('sentiment_score', 0), 3),
                'factual_accuracy': round(1 - original_metrics.get('hallucination_score', 0.5), 3),
                'diversity': round(original_metrics.get('diversity_score', 0.5), 3),
                'length': round(original_metrics.get('length_score', 0.5), 3)
            },
            'optimized_metrics': {
                'sentiment': round(optimized_metrics.get('sentiment_score', 0), 3),
                'factual_accuracy': round(1 - optimized_metrics.get('hallucination_score', 0.5), 3),
                'diversity': round(optimized_metrics.get('diversity_score', 0.5), 3),
                'length': round(optimized_metrics.get('length_score', 0.5), 3)
            },
            
            # Optimization details
            'optimization_history': optimization_history,
            'actions_applied': actions_applied,
            'learning_summary': result.get('learning_summary', {})
        }
        
        # Add to history
        add_to_history(response_data)
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Optimization failed: {e}")
        return jsonify({
            'error': 'Optimization service temporarily unavailable',
            'status': 'error'
        }), 500

@app.route('/history')
def get_history():
    """Get optimization history."""
    try:
        history = load_history()
        return jsonify(history)
    except Exception as e:
        print(f"Error loading history: {e}")
        return jsonify([])

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'optimizer_ready': optimizer is not None,
        'groq_ready': groq_client is not None
    })

if __name__ == '__main__':
    print("🚀 Starting A2C Prompt Optimization Demo...")
    print("📱 Web interface available at: http://localhost:5001")
    print("🔧 Debug mode: ON (pin will be shown in console)")
    app.run(host='0.0.0.0', port=5001, debug=True) 