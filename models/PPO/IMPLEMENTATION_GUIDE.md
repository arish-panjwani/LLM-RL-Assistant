# PPO Model Implementation Guide

## 🚀 Quick Start

This guide provides two implementation options for the PPO (Proximal Policy Optimization) model:

1. **Direct Model File Implementation** - Use the trained `.pth` file directly
2. **Docker Container Implementation** - Deploy as a containerized application

---

## 📋 Prerequisites

### System Requirements
- Python 3.8+ 
- 4GB+ RAM
- Internet connection (for API calls)
- GPU optional (CPU works fine)

### Required API Keys
You'll need these API keys in a `.env` file:
```env
GROQ_API_KEY=your_groq_api_key_here
WOLFRAM_APP_ID=your_wolfram_app_id_here
GOOGLE_API_KEY=your_google_api_key_here
```

---

## 🎯 Option 1: Direct Model File Implementation

### Step 1: Download Model Files
Download these files to your project:
- `saved_model/ppo_actor.pth` - The trained PPO model
- `utils.py` - Environment and prompt handling
- `model.py` - PPO agent implementation
- `requirements.txt` - Python dependencies

### Step 2: Install Dependencies
```bash
pip install torch sentence-transformers python-dotenv requests nltk
```

### Step 3: Basic Implementation
```python
import torch
from sentence_transformers import SentenceTransformer
from model import PPOAgent
from utils import PromptEnvironment
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

class PPOModel:
    def __init__(self, model_path="saved_model/ppo_actor.pth"):
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load encoder
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Create agent
        state_dim = self.encoder.get_sentence_embedding_dimension()
        action_dim = state_dim
        self.agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, device=self.device)
        
        # Load trained model
        if os.path.exists(model_path):
            self.agent.load(model_path)
            print("✅ Model loaded successfully")
        else:
            print("⚠️  No trained model found, using untrained agent")
        
        # Setup environment
        self.env = PromptEnvironment(self.encoder)
    
    def optimize_prompt(self, prompt_text):
        """Optimize a prompt using the PPO model"""
        try:
            # Set original prompt
            self.env.original_prompt = prompt_text
            
            # Encode prompt
            state = self.env.encode(prompt_text).unsqueeze(0).to(self.device)
            
            # Get action from agent
            action, _, _ = self.agent.select_action(state)
            action = torch.tensor(action, dtype=torch.float32).to(self.device)
            
            # Decode optimized prompt
            optimized_prompt = self.env.decode(action.squeeze())
            
            # Get LLM response
            llm_response = self.env.real_llm_response(optimized_prompt)
            
            return {
                'original': prompt_text,
                'optimized': optimized_prompt,
                'llm_response': llm_response,
                'success': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# Usage Example
if __name__ == "__main__":
    model = PPOModel()
    
    # Test optimization
    result = model.optimize_prompt("Explain quantum computing")
    if result['success']:
        print(f"Original: {result['original']}")
        print(f"Optimized: {result['optimized']}")
        print(f"Response: {result['llm_response']}")
    else:
        print(f"Error: {result['error']}")
```

### Step 4: Web API Implementation
```python
from flask import Flask, request, jsonify
from PPOModel import PPOModel

app = Flask(__name__)
model = PPOModel()

@app.route('/optimize', methods=['POST'])
def optimize():
    data = request.get_json()
    prompt = data.get('prompt', '')
    
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    result = model.optimize_prompt(prompt)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

---

## 🐳 Option 2: Docker Implementation

### Step 1: Create Dockerfile
```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files
COPY saved_model/ ./saved_model/
COPY utils.py .
COPY model.py .

# Copy application code
COPY app.py .

# Create .env file (you'll need to provide your API keys)
COPY .env .

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
```

### Step 2: Create docker-compose.yml
```yaml
version: '3.8'

services:
  ppo-model:
    build: .
    ports:
      - "5000:5000"
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
      - WOLFRAM_APP_ID=${WOLFRAM_APP_ID}
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
    volumes:
      - ./saved_model:/app/saved_model
    restart: unless-stopped
```

### Step 3: Build and Run
```bash
# Build the Docker image
docker build -t ppo-model .

# Run with docker-compose
docker-compose up -d

# Or run directly with Docker
docker run -p 5000:5000 --env-file .env ppo-model
```

---

## 🔧 API Endpoints

### POST /optimize
Optimize a prompt using the PPO model.

**Request:**
```json
{
  "prompt": "Explain machine learning to a beginner"
}
```

**Response:**
```json
{
  "success": true,
  "original": "Explain machine learning to a beginner",
  "optimized": "Please provide a clear and simple explanation of machine learning concepts suitable for beginners, including basic definitions and real-world examples.",
  "llm_response": "Machine learning is a subset of artificial intelligence..."
}
```

### GET /status
Check model status and health.

**Response:**
```json
{
  "model_loaded": true,
  "agent_loaded": true,
  "encoder_loaded": true,
  "env_loaded": true,
  "device": "cpu",
  "has_pretrained": true
}
```

---

## 📊 Model Architecture

### PPO Agent Components
- **Actor Network**: Generates optimized prompts
- **Critic Network**: Evaluates prompt quality
- **GAE (Generalized Advantage Estimation)**: Calculates advantages
- **Clipping**: Prevents large policy updates

### Prompt Optimization Process
1. **Encoding**: Convert text prompt to embedding vector
2. **Action Selection**: PPO agent generates optimized embedding
3. **Decoding**: Convert optimized embedding back to text
4. **LLM Response**: Get response from Groq API
5. **Reward Calculation**: Evaluate response quality

---

## 🔍 Troubleshooting

### Common Issues

**1. Model Not Loading**
```bash
# Check if model file exists
ls -la saved_model/ppo_actor.pth

# Verify file permissions
chmod 644 saved_model/ppo_actor.pth
```

**2. API Key Errors**
```bash
# Check environment variables
echo $GROQ_API_KEY
echo $WOLFRAM_APP_ID
echo $GOOGLE_API_KEY
```

**3. Memory Issues**
```bash
# Monitor memory usage
docker stats

# Increase Docker memory limit
docker run --memory=4g ppo-model
```

**4. Import Errors**
```bash
# Install missing dependencies
pip install torch sentence-transformers python-dotenv requests nltk

# Check Python path
python -c "import sys; print(sys.path)"
```

### Debug Endpoints
- `/api/debug` - Detailed system information
- `/api/test` - Test model functionality
- `/api/status` - Model status check

---

## 📈 Performance Optimization

### CPU Optimization
```python
# Use CPU efficiently
torch.set_num_threads(4)  # Adjust based on CPU cores
```

### Memory Optimization
```python
# Clear GPU memory if using CUDA
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

### Batch Processing
```python
def optimize_batch(prompts):
    """Optimize multiple prompts efficiently"""
    results = []
    for prompt in prompts:
        result = model.optimize_prompt(prompt)
        results.append(result)
    return results
```

---

## 🔐 Security Considerations

### API Key Security
- Store API keys in environment variables
- Never commit `.env` files to version control
- Use Docker secrets for production

### Input Validation
```python
def validate_prompt(prompt):
    """Validate input prompt"""
    if not prompt or len(prompt.strip()) == 0:
        raise ValueError("Empty prompt")
    if len(prompt) > 1000:
        raise ValueError("Prompt too long")
    return prompt.strip()
```

### Rate Limiting
```python
from flask_limiter import Limiter

limiter = Limiter(app)

@app.route('/optimize', methods=['POST'])
@limiter.limit("10 per minute")
def optimize():
    # Your optimization code here
    pass
```

---

## 📝 Production Deployment

### Environment Variables
```bash
# Production environment
export FLASK_ENV=production
export GROQ_API_KEY=your_production_key
export WOLFRAM_APP_ID=your_production_id
export GOOGLE_API_KEY=your_production_key
```

### Health Checks
```python
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model_manager.get_status()['model_loaded']
    })
```

### Logging
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

---

## 📞 Support

For implementation support:
1. Check the debug endpoints first
2. Review the troubleshooting section
3. Check model file integrity
4. Verify API key permissions

**Model Files Checklist:**
- [ ] `saved_model/ppo_actor.pth` exists
- [ ] `utils.py` is in the same directory
- [ ] `model.py` is in the same directory
- [ ] `.env` file with API keys
- [ ] All dependencies installed

---

## 🎉 Success Indicators

Your implementation is working correctly when:
- ✅ Model loads without errors
- ✅ `/api/status` returns `model_loaded: true`
- ✅ `/api/test` returns `success: true`
- ✅ Prompt optimization produces meaningful results
- ✅ LLM responses are generated successfully

**Example Success Response:**
```json
{
  "success": true,
  "original": "What is AI?",
  "optimized": "Please provide a comprehensive explanation of artificial intelligence, including its definition, types, applications, and current state of development.",
  "llm_response": "Artificial Intelligence (AI) refers to the simulation of human intelligence..."
}
```

---

*This implementation guide provides everything needed to deploy the PPO model in production environments. The model is designed to optimize prompts for better LLM responses using reinforcement learning techniques.* 