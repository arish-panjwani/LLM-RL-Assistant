# RL Prompt Optimizer - PPO Model

## 🎯 **Project Overview**
This is the **PPO (Proximal Policy Optimization)** component of the Intelligent Conversational Assistant with RL-Optimized Prompt Engineering.

## 🚀 **Quick Start for Group Members**

### **Option 1: Docker (Recommended)**
```bash
# 1. Clone the repository
git clone <your-repo-url>
cd rl_prompt_optimizer

# 2. Set up environment variables
cp .env.example .env
# Edit .env with your API keys:
# GROQ_API_KEY=your_groq_api_key
# GOOGLE_API_KEY=your_google_api_key
# WOLFRAM_APP_ID=your_wolfram_app_id

# 3. Run with Docker
docker-compose up --build

# 4. Access the API
curl http://localhost:8000/optimize_prompt -X POST \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is machine learning?"}'
```

### **Option 2: Local Development**
```bash
# 1. Create virtual environment
python -m venv rl_env
source rl_env/bin/activate  # On Windows: rl_env\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
export GROQ_API_KEY=your_groq_api_key

# 4. Train the model (demo mode - 500 iterations)
python main.py all

# 5. Start the server
python server/deployment_server.py
```

## 📊 **Model Information**

### **Training Configuration**
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Training Steps**: 500 (demo mode)
- **Environments**: 2 parallel environments
- **Model Path**: `./models/PPO_final.zip`

### **API Endpoints**
- `POST /optimize_prompt`: Optimize a prompt using the trained PPO model
- `GET /health`: Health check
- `GET /model_info`: Get model information

### **Integration with Streamlit/WebApp**
```python
import requests

def optimize_prompt(prompt_text):
    response = requests.post(
        "http://localhost:8000/optimize_prompt",
        json={"prompt": prompt_text}
    )
    return response.json()["optimized_prompt"]
```

## 🔄 **Complete Pipeline Flow**

### **1. User Input → Flask Server**
```python
# User sends: "What is machine learning?"
POST /optimize_prompt
{
    "prompt": "What is machine learning?"
}
```

### **2. Flask Server → PPO Model**
```python
# Flask server loads your trained PPO model
ppo_model = PPOModel(config)  # Loads ./models/ppo_final.zip

# PPO model optimizes the prompt
optimized_embedding = ppo_model.optimize_prompt(original_embedding)
```

### **3. PPO Model → Groq API**
```python
# Optimized prompt sent to Groq
groq_response = groq_client.get_response(optimized_prompt)
# Result: Much better, more specific response
```

## 🎯 **Three Optimization Goals (Assignment Requirements):**

### **1. Clarity & Specificity**
```python
# Your model learns to:
- Reduce ambiguity in prompts
- Add specific details and context
- Improve coherence between prompt and response
- Maximize cosine similarity with ideal responses
```

### **2. Response Relevance**
```python
# Your model learns to:
- Align prompts with user intent
- Minimize response rejection rates
- Optimize for positive user feedback
- Improve sentiment scores
```

### **3. Hallucination Reduction**
```python
# Your model learns to:
- Add fact-checking cues to prompts
- Request verification from reliable sources
- Minimize misinformation
- Penalize factually incorrect responses
```

## 🐳 **Docker Integration**

### **What Gets Dockerized:**
```
rl_prompt_optimizer/
├── models/ppo/ppo_model.py     ← PPO model wrapper
├── models/ppo/ppo_trainer.py   ← Training code
├── server/api_routes.py        ← Flask endpoints
├── utils/groq_client.py        ← Groq API client
├── config/config.py            ← Configuration
└── models/ppo_final.zip        ← Your trained model
```

### **Docker Build Process:**
```dockerfile
# Dockerfile copies your entire project
COPY . /app

# When container starts:
# 1. Loads trained PPO model from ./models/ppo_final.zip
# 2. Starts Flask server
# 3. Ready to accept requests
```

## 🚀 **API Endpoints Available:**

### **Main Optimization Endpoint:**
```bash
curl -X POST http://localhost:8000/optimize_prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is machine learning?"}'
```

**Response:**
```json
{
  "success": true,
  "original_prompt": "What is machine learning?",
  "optimized_prompt": "Please provide a comprehensive, accurate explanation of machine learning, including its definition, applications, and current developments, with specific examples and reliable sources.",
  "llm_response": "Machine learning is a subset of artificial intelligence...",
  "model_used": "PPO",
  "metrics": {
    "clarity_score": 0.85,
    "relevance_score": 0.92,
    "hallucination_penalty": 0.15
  }
}
```

## ✅ **What Your Group Gets:**

### **For Streamlit Integration:**
```python
import requests

def optimize_and_get_response(user_input):
    # Send to your Flask server
    response = requests.post(
        "http://localhost:8000/optimize_prompt",
        json={"prompt": user_input}
    )
    
    result = response.json()
    return {
        "original": result["original_prompt"],
        "optimized": result["optimized_prompt"], 
        "response": result["llm_response"]
    }

# Usage in Streamlit
user_input = st.text_input("Ask a question:")
if user_input:
    result = optimize_and_get_response(user_input)
    st.write(f"Original: {result['original']}")
    st.write(f"Optimized: {result['optimized']}")
    st.write(f"Response: {result['response']}")
```

### **For Custom WebApp:**
```javascript
// Frontend JavaScript
async function optimizePrompt(prompt) {
    const response = await fetch('/optimize_prompt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: prompt })
    });
    return await response.json();
}
```

## 🎯 **Complete Pipeline:**

```
User Input
    ↓
Flask Server (Docker)
    ↓
PPO Model (./models/ppo_final.zip)
    ↓
Optimized Prompt
    ↓
Groq API (LLM)
    ↓
Enhanced Response
    ↓
User (via Streamlit/WebApp)
```

## 🔧 **Configuration**

### **Environment Variables**
- `GROQ_API_KEY`: Your Groq API key (required)
- `GOOGLE_API_KEY`: Google API key for fact verification (optional)
- `WOLFRAM_APP_ID`: Wolfram Alpha app ID for fact verification (optional)

### **Model Parameters**
- **Embedding Dimension**: 384
- **Action Dimension**: 384
- **Learning Rate**: 3e-4
- **Batch Size**: 32 (demo mode)
- **Epochs**: 4 (demo mode)

## 📁 **Project Structure**
```
rl_prompt_optimizer/
├── models/ppo/           # PPO model implementation
├── environment/          # RL environment
├── utils/               # Utilities (Groq client, etc.)
├── config/              # Configuration
├── server/              # API server
├── training/            # Training scripts
├── data/                # Training data
├── logs/                # Training logs
├── demo/                # Demo implementations
├── Dockerfile           # Docker configuration
├── docker-compose.yaml  # Docker compose
└── requirements.txt     # Dependencies
```

## 🎓 **For School Demo**
- **Training Time**: ~2-3 minutes
- **API Calls**: Minimal (cached responses)
- **Model Size**: ~50MB
- **Memory Usage**: ~2GB RAM

## 🔗 **Integration Points**
This PPO model is designed to work with:
- **Streamlit**: Web interface for testing
- **Custom WebApp**: REST API integration
- **Raspberry Pi**: Lightweight deployment
- **Smartphone Apps**: API consumption

## 📝 **Notes for Group Members**
1. The model is trained for **demo purposes** (500 iterations)
2. For production, increase `TOTAL_TIMESTEPS` to 10,000+
3. API rate limiting is implemented to avoid costs
4. Model saves automatically to `./models/PPO_final.zip`
5. All dependencies are included in Docker image

## 🐛 **Troubleshooting**
- **Model not found**: Run training first with `python main.py all`
- **API errors**: Check your API keys in `.env` file
- **Docker issues**: Ensure Docker is installed and running
- **Memory issues**: Reduce `N_ENVS` in config if needed

## 🚀 **Demo Folder**
Check the `demo/` folder for:
- Streamlit implementation examples
- WebApp integration code
- Docker deployment guides
- Testing scripts