#!/usr/bin/env python3
"""
Flask WebApp for PPO Model Testing
"""

import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify, session
import torch
from sentence_transformers import SentenceTransformer
from utils import PromptEnvironment
from model import PPOAgent
import json
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'ppo_demo_secret_key'

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
        """Initialize the PPO model - called at startup"""
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
            
            print("🔧 Setting up PPO agent...")
            state_dim = self.encoder.get_sentence_embedding_dimension()
            action_dim = state_dim
            print(f"✅ State/Action dimensions: {state_dim}")
            
            self.agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, device=self.device)
            print("✅ PPO agent created")
            
            # Try to load pre-trained model
            model_path = "../saved_model/ppo_actor.pth"
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
            'has_pretrained': os.path.exists("../saved_model/ppo_actor.pth"),
            'initialized': self._initialized
        }
        return status
    
    def optimize_prompt(self, prompt_text):
        """Optimize a prompt using the PPO model"""
        # Check model status
        status = self.get_status()
        if not status['model_loaded']:
            return None, f"Model not loaded. Status: {status}"
        
        if self.agent is None or self.env is None:
            return None, f"Model components missing. Agent: {self.agent is not None}, Env: {self.env is not None}"
        
        try:
            print(f"🔧 Optimizing prompt: {prompt_text[:50]}...")
            self.env.original_prompt = prompt_text
            state = self.env.encode(prompt_text).unsqueeze(0).to(self.device)
            print("✅ Prompt encoded")
            
            action, _, _ = self.agent.select_action(state)
            action = torch.tensor(action, dtype=torch.float32).to(self.device)
            print("✅ Action selected")
            
            optimized_prompt = self.env.decode(action.squeeze())
            print(f"✅ Optimized prompt: {optimized_prompt[:50]}...")
            
            # Get LLM response
            print("🔧 Getting LLM response...")
            llm_response = self.env.real_llm_response(optimized_prompt)
            print("✅ LLM response received")
            
            return optimized_prompt, llm_response
        except Exception as e:
            print(f"❌ Error optimizing prompt: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, f"Error: {str(e)}"

# Create global model manager instance
model_manager = ModelManager()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/optimize', methods=['POST'])
def api_optimize():
    """API endpoint for prompt optimization"""
    data = request.get_json()
    prompt = data.get('prompt', '').strip()
    
    if not prompt:
        return jsonify({'error': 'Please provide a prompt'})
    
    optimized, llm_response = model_manager.optimize_prompt(prompt)
    
    if optimized is None:
        return jsonify({'error': llm_response})
    
    # Store in session for feedback
    session['last_optimization'] = {
        'original': prompt,
        'optimized': optimized,
        'llm_response': llm_response,
        'timestamp': datetime.now().isoformat()
    }
    
    return jsonify({
        'original': prompt,
        'optimized': optimized,
        'llm_response': llm_response,
        'success': True
    })

@app.route('/api/feedback', methods=['POST'])
def api_feedback():
    """API endpoint for user feedback"""
    data = request.get_json()
    feedback = data.get('feedback', '').lower()
    
    if feedback not in ['satisfied', 'not_satisfied']:
        return jsonify({'error': 'Invalid feedback'})
    
    last_opt = session.get('last_optimization')
    if not last_opt:
        return jsonify({'error': 'No optimization to provide feedback on'})
    
    # Store feedback
    if 'feedback_history' not in session:
        session['feedback_history'] = []
    
    session['feedback_history'].append({
        'original': last_opt['original'],
        'optimized': last_opt['optimized'],
        'llm_response': last_opt['llm_response'],
        'satisfied': feedback == 'satisfied',
        'timestamp': last_opt['timestamp']
    })
    
    # Calculate statistics
    history = session['feedback_history']
    total = len(history)
    satisfied = sum(1 for h in history if h['satisfied'])
    satisfaction_rate = (satisfied / total * 100) if total > 0 else 0
    
    return jsonify({
        'success': True,
        'statistics': {
            'total_feedback': total,
            'satisfied_count': satisfied,
            'satisfaction_rate': round(satisfaction_rate, 1)
        }
    })

@app.route('/api/statistics')
def api_statistics():
    """API endpoint for feedback statistics"""
    history = session.get('feedback_history', [])
    total = len(history)
    satisfied = sum(1 for h in history if h['satisfied'])
    satisfaction_rate = (satisfied / total * 100) if total > 0 else 0
    
    return jsonify({
        'total_feedback': total,
        'satisfied_count': satisfied,
        'satisfaction_rate': round(satisfaction_rate, 1),
        'recent_feedback': history[-5:] if history else []
    })

@app.route('/api/status')
def api_status():
    """API endpoint for model status"""
    status = model_manager.get_status()
    return jsonify(status)

@app.route('/api/reload')
def api_reload():
    """API endpoint to reload the model"""
    print("🔄 Reloading model...")
    success = model_manager.initialize()
    status = model_manager.get_status()
    return jsonify({
        'success': success,
        'status': status
    })

@app.route('/api/test')
def api_test():
    """API endpoint to test model functionality"""
    try:
        # Test basic model access
        status = model_manager.get_status()
        
        if not status['model_loaded']:
            return jsonify({
                'success': False,
                'error': 'Model not loaded',
                'status': status
            })
        
        # Test optimization with a simple prompt
        test_prompt = "Hello world"
        optimized, llm_response = model_manager.optimize_prompt(test_prompt)
        
        if optimized is None:
            return jsonify({
                'success': False,
                'error': llm_response,
                'status': status
            })
        
        return jsonify({
            'success': True,
            'test_prompt': test_prompt,
            'optimized_prompt': optimized,
            'llm_response': llm_response,
            'status': status
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'status': model_manager.get_status()
        })

@app.route('/api/debug')
def api_debug():
    """API endpoint for detailed debugging information"""
    import traceback
    
    debug_info = {
        'model_status': model_manager.get_status(),
        'python_path': sys.path,
        'current_directory': os.getcwd(),
        'parent_directory': os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'env_file_exists': os.path.exists("../.env"),
        'model_file_exists': os.path.exists("../saved_model/ppo_actor.pth"),
        'utils_exists': os.path.exists("../utils.py"),
        'model_py_exists': os.path.exists("../model.py")
    }
    
    try:
        # Test imports
        import torch
        debug_info['torch_available'] = True
        debug_info['torch_version'] = torch.__version__
    except ImportError as e:
        debug_info['torch_available'] = False
        debug_info['torch_error'] = str(e)
    
    try:
        from sentence_transformers import SentenceTransformer
        debug_info['sentence_transformers_available'] = True
    except ImportError as e:
        debug_info['sentence_transformers_available'] = False
        debug_info['sentence_transformers_error'] = str(e)
    
    try:
        from utils import PromptEnvironment
        debug_info['utils_import_success'] = True
    except ImportError as e:
        debug_info['utils_import_success'] = False
        debug_info['utils_import_error'] = str(e)
    
    try:
        from model import PPOAgent
        debug_info['model_import_success'] = True
    except ImportError as e:
        debug_info['model_import_success'] = False
        debug_info['model_import_error'] = str(e)
    
    return jsonify(debug_info)

if __name__ == '__main__':
    # Initialize model at startup
    print("🚀 Initializing PPO Model...")
    model_loaded = model_manager.initialize()
    
    if model_loaded:
        print("✅ Model initialized successfully!")
        status = model_manager.get_status()
        print(f"   Agent: {status['agent_loaded']}")
        print(f"   Environment: {status['env_loaded']}")
        print(f"   Device: {status['device']}")
    else:
        print("⚠️  Model initialization failed!")
    
    print("🌐 Starting Flask webapp...")
    app.run(debug=False, host='0.0.0.0', port=5000) 