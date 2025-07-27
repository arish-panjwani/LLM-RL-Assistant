# Demo Implementation Guide

## 🎯 **Overview**
This demo folder contains everything the group needs to deploy and integrate the PPO model with Streamlit, custom webapps, and other interfaces.

## 📁 **Demo Folder Structure**
```
demo/
├── README.md                    # This file - deployment guide
├── streamlit_app.py            # Complete Streamlit implementation
├── webapp_example/             # Custom webapp example
│   ├── index.html
│   ├── style.css
│   └── script.js
├── docker_deployment.md        # Docker deployment guide
├── api_testing.py              # API testing script
├── integration_examples.py     # Integration code examples
└── requirements_demo.txt       # Demo-specific requirements
```

## 🚀 **Quick Start**

### **1. Docker Deployment (Recommended)**
```bash
# Navigate to main project directory
cd rl_prompt_optimizer

# Build and run with Docker
docker-compose up --build

# Test the API
python demo/api_testing.py
```

### **2. Streamlit Demo**
```bash
# Install demo requirements
pip install -r demo/requirements_demo.txt

# Run Streamlit app
streamlit run demo/streamlit_app.py
```

### **3. Custom WebApp Demo**
```bash
# Serve the webapp files
cd demo/webapp_example
python -m http.server 8080

# Open http://localhost:8080 in browser
```

## 🐳 **Docker Deployment Guide**

### **Step 1: Prepare the Environment**
```bash
# 1. Ensure Docker is installed
docker --version

# 2. Set up environment variables
cp .env.example .env
# Edit .env with your API keys:
# GROQ_API_KEY=your_groq_api_key
# GOOGLE_API_KEY=your_google_api_key
# WOLFRAM_APP_ID=your_wolfram_app_id
```

### **Step 2: Build and Run**
```bash
# 1. Build the Docker image
docker build -t rl-prompt-optimizer .

# 2. Run with docker-compose (recommended)
docker-compose up --build

# 3. Or run manually
docker run -p 8000:8000 \
  -e GROQ_API_KEY=your_api_key \
  -e GOOGLE_API_KEY=your_google_key \
  -e WOLFRAM_APP_ID=your_wolfram_id \
  rl-prompt-optimizer
```

### **Step 3: Verify Deployment**
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test optimization endpoint
curl -X POST http://localhost:8000/optimize_prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is AI?"}'
```

## 📊 **What Gets Dockerized**

### **Essential Files Included:**
```
rl_prompt_optimizer/
├── config/config.py            # Configuration
├── models/ppo/                 # PPO model code
│   ├── ppo_model.py
│   └── ppo_trainer.py
├── server/                     # Flask server
│   ├── api_routes.py
│   └── deployment_server.py
├── utils/                      # Utilities
│   ├── groq_client.py
│   └── evaluation.py
├── environment/                # RL environment
│   ├── prompt_env.py
│   └── reward_calculator.py
├── training/                   # Training scripts
├── data/                       # Data loading
├── main.py                     # Entry point
├── requirements.txt            # Dependencies
├── Dockerfile                  # Docker config
└── docker-compose.yaml         # Docker compose
```

### **Files Excluded (via .dockerignore):**
```
rl_env/                         # Virtual environment
logs/                           # Generated logs
tensorboard_logs/               # Generated logs
test/                           # Test files
*.ipynb                         # Jupyter notebooks
.env                            # Environment variables
```

## 🔧 **Docker Configuration**

### **Dockerfile Breakdown:**
```dockerfile
# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models logs data tensorboard_logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "main.py"]
```

### **Docker Compose Configuration:**
```yaml
version: '3.8'
services:
  rl-prompt-optimizer:
    build: .
    ports:
      - "8000:8000"
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - WOLFRAM_APP_ID=${WOLFRAM_APP_ID}
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
      - ./data:/app/data
    command: python server/deployment_server.py
```

## 🎯 **API Integration Examples**

### **Python Integration:**
```python
import requests

def optimize_prompt(prompt_text):
    """Send prompt to PPO model for optimization"""
    response = requests.post(
        "http://localhost:8000/optimize_prompt",
        json={"prompt": prompt_text}
    )
    return response.json()

# Usage
result = optimize_prompt("What is machine learning?")
print(f"Original: {result['original_prompt']}")
print(f"Optimized: {result['optimized_prompt']}")
print(f"Response: {result['llm_response']}")
```

### **JavaScript Integration:**
```javascript
async function optimizePrompt(prompt) {
    const response = await fetch('http://localhost:8000/optimize_prompt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: prompt })
    });
    return await response.json();
}

// Usage
optimizePrompt("What is AI?").then(result => {
    console.log("Optimized prompt:", result.optimized_prompt);
    console.log("LLM response:", result.llm_response);
});
```

### **cURL Integration:**
```bash
curl -X POST http://localhost:8000/optimize_prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum computing"}'
```

## 🧪 **Testing the Deployment**

### **1. Health Check:**
```bash
curl http://localhost:8000/health
# Expected: {"status": "healthy", "model_loaded": true}
```

### **2. Model Info:**
```bash
curl http://localhost:8000/model_info
# Expected: Model information and status
```

### **3. Prompt Optimization:**
```bash
curl -X POST http://localhost:8000/optimize_prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is machine learning?"}'
```

### **4. Batch Testing:**
```bash
python demo/api_testing.py
```

## 🔄 **Complete Pipeline Flow**

```
1. User Input
   ↓
2. Flask Server (Docker Container)
   ↓
3. PPO Model (./models/ppo_final.zip)
   ↓
4. Optimized Prompt
   ↓
5. Groq API (LLM)
   ↓
6. Enhanced Response
   ↓
7. User (via Streamlit/WebApp)
```

## 🎓 **For School Demo**

### **Demo Script:**
1. **Start the system**: `docker-compose up --build`
2. **Show training**: Point out the 500 iterations and fast training
3. **Test API**: Use `curl` or the testing script
4. **Show Streamlit**: Run `streamlit run demo/streamlit_app.py`
5. **Explain the pipeline**: Walk through the complete flow

### **Key Points to Highlight:**
- ✅ **Fast Training**: 2-3 minutes for 500 iterations
- ✅ **Real API Integration**: Uses actual Groq API
- ✅ **Complete Pipeline**: Training → Optimization → Response
- ✅ **Docker Ready**: Easy deployment for group members
- ✅ **Multiple Interfaces**: Streamlit, WebApp, API

## 🐛 **Troubleshooting**

### **Common Issues:**

1. **Docker Build Fails:**
   ```bash
   # Clean and rebuild
   docker system prune -a
   docker-compose build --no-cache
   ```

2. **API Connection Refused:**
   ```bash
   # Check if container is running
   docker ps
   # Check logs
   docker-compose logs
   ```

3. **Model Not Found:**
   ```bash
   # Train the model first
   python main.py all
   # Or check model path
   ls -la models/
   ```

4. **API Key Issues:**
   ```bash
   # Check environment variables
   docker-compose config
   # Verify .env file
   cat .env
   ```

## 📝 **Next Steps for the Group**

1. **Deploy with Docker**: Use the docker-compose setup
2. **Integrate with Streamlit**: Use the provided Streamlit app
3. **Build Custom WebApp**: Use the webapp example as template
4. **Test thoroughly**: Use the testing scripts
5. **Scale as needed**: Multiple containers for production

## 🎉 **Success Criteria**

Deployment is successful when:
- ✅ Docker container starts without errors
- ✅ Health endpoint returns `{"status": "healthy"}`
- ✅ Prompt optimization endpoint works
- ✅ Streamlit app can connect to the API
- ✅ Custom webapp can make requests
- ✅ All group members can access the system

**Ready to hand over a complete, working PPO model!** 🚀 