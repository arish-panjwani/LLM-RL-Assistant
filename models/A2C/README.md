# 🤖 A2C (Actor-Critic) Reinforcement Learning Model for Dynamic Prompt Optimization

**Kauthara's Assignment: Intelligent Conversational Assistant with RL-Optimized Prompt Engineering**

## 🎯 Assignment Overview

This system implements an **A2C (Actor-Critic) reinforcement learning model** that dynamically optimizes prompts for better LLM responses. The model learns from:
- **Human feedback** (primary reward - 70% weight)
- **Groq API self-evaluation** (secondary reward - 30% weight)
- **Real-time learning** from each interaction

## ✨ Key Features

- **🤖 A2C Reinforcement Learning**: Dynamic prompt optimization with real-time learning
- **🔗 Groq API Integration**: Real LLM responses and self-evaluation
- **👥 Human Feedback System**: Interactive rating (Good/Okay/Bad) for responses
- **📊 Comprehensive Metrics**: Cosine similarity, sentiment analysis, factual verification
- **🌐 Domain-Agnostic**: Works across all topics and domains
- **⚡ Real-Time Learning**: Model updates immediately after each feedback
- **📈 Performance Tracking**: Persistent optimization history and metrics

## 🏗️ Project Structure

```
models/A2C/
├── src/
│   ├── models/           # A2C model & prompt optimizer
│   ├── utils/            # APIs, evaluation metrics, config
│   └── training/         # RL environment & trainer
├── demo/                 # Flask web interface for testing
├── scripts/              # Training & evaluation tools
├── config/               # Configuration files
├── data/                 # Models & optimization history
└── Dockerfile           # Production deployment
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Activate conda environment
conda activate a2c_env

# Set API key (optional - system works without it)
export GROQ_API_KEY="your_key_here"
# Or create .env file with: GROQ_API_KEY=your_key_here
```

### 2. Install Dependencies
```bash
pip install -r model_requirements.txt
```

### 3. Test the System
```bash
# Terminal-based feedback system
python scripts/test_feedback.py

# Web demo interface
cd demo
python app.py
# Visit http://localhost:5001
```

## 🎯 How It Works

### Learning Process:
1. **User Input**: Enter a prompt
2. **A2C Optimization**: Model modifies the prompt for better results
3. **LLM Response**: Groq API generates response (or mock if no API key)
4. **Human Feedback**: Rate the response (Good/Okay/Bad)
5. **Real-Time Learning**: Model learns from your feedback immediately
6. **Performance Tracking**: All interactions saved to history

### Reward System:
- **70% Human Feedback**: Your rating is the primary learning signal
- **30% Automated Metrics**: Cosine similarity, sentiment, factual accuracy
- **Dynamic Learning**: Model adapts to your preferences over time

## 📊 Performance & Evaluation

### Metrics Tracked:
- **Cosine Similarity**: Response relevance to original prompt
- **Sentiment Analysis**: Emotional tone of responses
- **Factual Accuracy**: Verification against external sources
- **Lexical Diversity**: Response variety and quality
- **Human Feedback**: Direct user ratings

### Learning Verification:
```bash
# Quick performance check after 10+ feedback samples
python scripts/quick_performance_check.py

# Full batch retraining with all collected data
python scripts/batch_retrain.py
```

## 🎯 Assignment Requirements Met

### ✅ Core Requirements:
- **RL Model**: A2C implementation with actor-critic architecture
- **Groq API Integration**: Real LLM responses and self-evaluation
- **Dynamic Learning**: Real-time adaptation from user feedback
- **Evaluation Metrics**: All required metrics implemented
- **Domain-Agnostic**: Works across all topics and domains

### ✅ Advanced Features:
- **Human Feedback Integration**: Primary reward mechanism
- **Persistent Learning**: History tracking and batch retraining
- **Team Integration**: Clear handoff documentation

## 👥 Team Handoff

### For Ujju (Prompt Engineer):
- **Primary File**: `src/models/prompt_optimizer.py`
- **Integration**: Use `PromptOptimizer` class for prompt optimization
- **Example**: See `demo/TEAM_HANDOFF.md` for integration code

### For Thejaswi (Data Analyst):
- **Primary File**: `scripts/metrics_analysis.py`
- **Data Source**: `demo/optimization_history.json`
- **Analysis**: Extract metrics from optimization history

### For Other Team Members:
- **Arish (System Integration)**: `src/utils/groq_client.p
- **Kanika (Testing)**: Use `scripts/test_feedback.py`
- **Mueez (Deployment)**: Use `Dockerfile`

## 🔧 API Setup (Optional)

For real LLM responses, see `API_SETUP.md` for detailed instructions on:
- Groq API key setup
- Google API for fact verification
- Wikipedia API integration

## 📝 Files Overview

### Core A2C Implementation:
- `src/models/a2c_model.py` - Neural network architecture
- `src/models/prompt_optimizer.py` - Main optimization logic
- `src/training/trainer.py` - Training and learning algorithms
- `src/training/environment.py` - RL environment

### Evaluation & Metrics:
- `src/utils/evaluation_metrics.py` - All evaluation metrics
- `src/utils/groq_client.py` - Groq API integration
- `scripts/metrics_analysis.py` - Data analysis tools

`
### Testing & Demo:
- `demo/app.py` - Flask web interface
- `scripts/test_feedback.py` - Terminal feedback system
- `scripts/quick_performance_check.py` - Performance evaluation

## 🎉 Status: Assignment Complete

**✅ Kauthara's A2C Model**: Fully implemented and tested
**✅ Real-Time Learning**: Working with human feedback
**✅ Team Integration**: Ready for handoff

---

**🚀 Ready for Team Integration - A2C Model Successfully Implemented!** 
