#!/usr/bin/env python3
"""
Flask WebApp for A2C Model Testing
"""

import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify, session
import torch
from sentence_transformers import SentenceTransformer
from utils import PromptEnvironment
from model import A2CAgent
import json
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'a2c_demo_secret_key'

# Global variables for model - use a class to persist across restarts
class ModelManager:
    def __init__(self):
        self.agent = None
        self.encoder = None
        self.device = None
        self.env = None
        self.model_loaded = False
        self._initialized = False
    
    def initialize(self):
        """Initialize the A2C model - called at startup"""
        if self._initialized:
            print("✅ Model already initialized")
            return self.model_loaded
        
        try:
            print("🔧 Setting up device...")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"✅ Using device: {self.device}")
            
            print("🔧 Loading sentence transformer...")
            self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
            print("✅ Sentence transformer loaded")
            
            print("🔧 Setting up A2C agent...")
            state_dim = self.encoder.get_sentence_embedding_dimension()
            action_dim = state_dim
            print(f"✅ State/Action dimensions: {state_dim}")
            
            self.agent = A2CAgent(state_dim=state_dim, action_dim=action_dim, device=self.device)
            print("✅ A2C agent created")
            
            # Try to load pre-trained model
            model_path = "../saved_model/a2c_actor.pth"
            if os.path.exists(model_path):
                print(f"🔧 Loading pre-trained model from: {model_path}")
                self.agent.load(model_path)
                print("✅ Loaded pre-trained model")
            else:
                print("⚠️  No pre-trained model found. Using untrained model.")
                print(f"   Expected path: {os.path.abspath(model_path)}")
            
            print("🔧 Setting up environment...")
            self.env = PromptEnvironment(self.encoder)
            print("✅ Environment created")
            
            # Set model as loaded
            self.model_loaded = True
            self._initialized = True
            print("✅ Model loading completed successfully!")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
            self.model_loaded = False
            return False
    
    def get_status(self):
        """Get current model status"""
        status = {
            'model_loaded': self.model_loaded,
            'agent_loaded': self.agent is not None,
            'encoder_loaded': self.encoder is not None,
            'env_loaded': self.env is not None,
            'device': str(self.device) if self.device else "Not available",
            'has_pretrained': os.path.exists("../saved_model/a2c_actor.pth"),
            'initialized': self._initialized
        }
        return status
    
    def optimize_prompt(self, prompt_text):
        """Optimize a prompt using the A2C model"""
        # Check model status
        status = self.get_status()
        if not status['model_loaded']:
            return {
                'success': False,
                'error': 'Model not loaded. Please check the server logs.'
            }
        
        try:
            # Set the original prompt
            self.env.original_prompt = prompt_text
            
            # Encode the prompt
            state = self.env.encode(prompt_text).unsqueeze(0).to(self.device)
            
            # Get A2C action
            action, log_prob, value = self.agent.select_action(state)
            action = torch.tensor(action, dtype=torch.float32).to(self.device)
            
            # Decode the action to get optimized prompt
            optimized_prompt = self.env.decode(action.squeeze())
            
            # Get LLM response
            response = self.env.real_llm_response(optimized_prompt)
            
            # Calculate reward
            reward = self.env.calculate_reward_with_feedback(prompt_text, optimized_prompt, response)
            
            return {
                'success': True,
                'original': prompt_text,
                'optimized': optimized_prompt,
                'llm_response': response,
                'reward': float(reward),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error during optimization: {str(e)}'
            }

# Initialize model manager
model_manager = ModelManager()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/optimize', methods=['POST'])
def api_optimize():
    """API endpoint for prompt optimization"""
    try:
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({
                'success': False,
                'error': 'No prompt provided'
            }), 400
        
        prompt = data['prompt'].strip()
        if not prompt:
            return jsonify({
                'success': False,
                'error': 'Empty prompt provided'
            }), 400
        
        # Optimize the prompt
        result = model_manager.optimize_prompt(prompt)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/api/feedback', methods=['POST'])
def api_feedback():
    """API endpoint for user feedback"""
    try:
        data = request.get_json()
        if not data or 'satisfied' not in data:
            return jsonify({
                'success': False,
                'error': 'No feedback provided'
            }), 400
        
        satisfied = data['satisfied']
        original = data.get('original', '')
        optimized = data.get('optimized', '')
        response = data.get('response', '')
        
        # Store feedback
        if model_manager.env:
            model_manager.env.user_feedback_history.append({
                'original': original,
                'refined': optimized,
                'response': response,
                'satisfied': satisfied,
                'timestamp': len(model_manager.env.user_feedback_history)
            })
        
        return jsonify({
            'success': True,
            'message': 'Feedback recorded successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/api/statistics')
def api_statistics():
    """API endpoint for feedback statistics"""
    try:
        if model_manager.env and model_manager.env.user_feedback_history:
            stats = model_manager.env.get_feedback_statistics()
            if isinstance(stats, dict):
                return jsonify({
                    'success': True,
                    'statistics': stats
                })
            else:
                return jsonify({
                    'success': True,
                    'statistics': {'message': 'No feedback collected yet'}
                })
        else:
            return jsonify({
                'success': True,
                'statistics': {'message': 'No feedback collected yet'}
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/api/status')
def api_status():
    """API endpoint for model status"""
    try:
        status = model_manager.get_status()
        return jsonify({
            'success': True,
            'status': status
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/api/reload')
def api_reload():
    """API endpoint to reload the model"""
    try:
        model_manager._initialized = False
        success = model_manager.initialize()
        return jsonify({
            'success': success,
            'message': 'Model reloaded successfully' if success else 'Failed to reload model'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/api/test')
def api_test():
    """API endpoint for testing the model"""
    try:
        test_prompt = "Hello world"
        result = model_manager.optimize_prompt(test_prompt)
        
        if result['success']:
            return jsonify({
                'success': True,
                'test_result': result,
                'message': 'Model test successful'
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Unknown error')
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/api/debug')
def api_debug():
    """API endpoint for debugging information"""
    try:
        debug_info = {
            'model_status': model_manager.get_status(),
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'device': str(model_manager.device) if model_manager.device else 'Not available',
            'working_directory': os.getcwd(),
            'files_in_directory': os.listdir('.'),
            'model_files': {
                'model_py': os.path.exists('../model.py'),
                'utils_py': os.path.exists('../utils.py'),
                'saved_model_dir': os.path.exists('../saved_model'),
                'a2c_actor_pth': os.path.exists('../saved_model/a2c_actor.pth')
            }
        }
        
        return jsonify({
            'success': True,
            'debug_info': debug_info
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Initialize model on startup
    print("🚀 Starting A2C WebApp...")
    model_manager.initialize()
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True) 