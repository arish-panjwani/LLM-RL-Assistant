# A2C (Actor-Critic) Model for Prompt Optimization

This is an implementation of an Actor-Critic (A2C) reinforcement learning model for optimizing prompts in conversational AI systems. The model is designed to dynamically improve prompt engineering using reinforcement learning techniques.

## 🎯 Overview

The A2C model is part of an intelligent conversational assistant that:
- Uses Raspberry Pi as a central processing unit
- Leverages smartphones for user input/output
- Uses Groq APIs to access large language models
- Implements Actor-Critic reinforcement learning to optimize prompts dynamically

## 🏗️ Architecture

### Core Components

1. **Actor Network**: Generates optimized prompts based on input state
2. **Critic Network**: Evaluates the value of states to guide the actor
3. **Environment**: Manages prompt optimization and reward calculation
4. **Buffer**: Stores experience for training

### Key Features

- **Dynamic Prompt Optimization**: Modifies user queries before sending to Groq APIs
- **Multi-objective Reward Function**: Optimizes for clarity, specificity, and user satisfaction
- **User Feedback Integration**: Incorporates explicit user feedback into training
- **Hallucination Detection**: Penalizes factually incorrect responses
- **Sentiment Analysis**: Uses VADER sentiment analysis for response evaluation

## 📁 File Structure

```
A2C/
├── model.py              # A2C agent implementation
├── utils.py              # Environment and utilities
├── main.py               # Training script
├── interactive_inference.py  # Interactive testing
├── deploy.py             # Deployment script
├── deploy_model.py       # Quick deployment utility
├── demo/                 # Web application
│   ├── app.py           # Flask webapp
│   ├── run_demo.py      # Demo runner
│   ├── requirements.txt # Dependencies
│   └── templates/       # HTML templates
│       └── index.html   # Web interface
└── saved_model/         # Trained model storage
    └── a2c_actor.pth    # Saved actor weights
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install torch sentence-transformers flask openai python-dotenv nltk scikit-learn numpy requests

# Or install from requirements
cd demo
pip install -r requirements.txt
```

### 2. Environment Setup

Create a `.env` file with your API keys:

```env
GROQ_API_KEY=your_groq_api_key_here
WOLFRAM_APP_ID=your_wolfram_app_id_here
GOOGLE_API_KEY=your_google_api_key_here
```

### 3. Training

```bash
# Automated training
python main.py train

# Interactive training with feedback
python main.py interactive

# Or run interactively
python main.py
```

### 4. Testing

```bash
# Interactive inference
python interactive_inference.py

# Web demo
cd demo
python run_demo.py
```

### 5. Deployment

```bash
# Quick deployment check
python deploy_model.py

# Deploy model
python deploy.py
```

## 🧠 Model Details

### A2C Algorithm

The Actor-Critic algorithm combines:
- **Policy Gradient** (Actor): Learns to select actions (prompt modifications)
- **Value Function** (Critic): Learns to evaluate state values

### Reward Function

The reward function optimizes multiple objectives:

```
R = λ₁ × cosine_similarity(response_variations) 
    - λ₂ × redundancy_penalty 
    + λ₃ × Groq_rating 
    + α × user_rating 
    + β × sentiment_score 
    - γ × hallucination_score
```

Where:
- **λ₁, λ₂, λ₃**: Weighting factors for different components
- **α, β**: User feedback and sentiment weights
- **γ**: Hallucination penalty weight

### Network Architecture

- **Actor**: 3-layer neural network (256 hidden units)
- **Critic**: 3-layer neural network (256 hidden units)
- **State/Action Space**: Sentence embedding dimensions (384 for all-MiniLM-L6-v2)

## 📊 Training Modes

### 1. Automated Training
- Runs without user intervention
- Uses predefined prompts and simulated feedback
- Good for initial model training

### 2. Interactive Training
- Collects real user feedback
- Adapts to user preferences
- Improves model performance over time

### 3. Deployment Mode
- Real-time inference only
- No training updates
- Optimized for production use

## 🌐 Web Interface

The web demo provides:
- Real-time prompt optimization
- User feedback collection
- Performance statistics
- Model status monitoring

Access at: `http://localhost:5000`

## 📈 Performance Metrics

The model tracks:
- **User Satisfaction Rate**: Percentage of satisfied responses
- **Reward Scores**: Average reward per optimization
- **Feedback Statistics**: Detailed user feedback analysis
- **Model Performance**: Training and inference metrics

## 🔧 Configuration

### Hyperparameters

```python
# A2C specific parameters
gamma = 0.99              # Discount factor
value_coef = 0.5          # Value function coefficient
entropy_coef = 0.01       # Entropy bonus coefficient
max_grad_norm = 0.5       # Gradient clipping

# Training parameters
learning_rate_actor = 3e-4
learning_rate_critic = 1e-3
buffer_size = 10000
batch_size = 64
```

### Reward Weights

```python
λ1 = 1.0    # Cosine similarity weight
λ2 = 0.5    # Redundancy penalty weight
λ3 = 1.0    # Clarity rating weight
γ = 2.0     # Hallucination penalty weight
```

## 🐛 Troubleshooting

### Common Issues

1. **Missing API Keys**: Ensure all required API keys are in `.env`
2. **Model Not Loading**: Check if `saved_model/a2c_actor.pth` exists
3. **Dependencies**: Install all required packages
4. **CUDA Issues**: Model works on CPU if CUDA unavailable

### Debug Mode

```bash
# Check model status
curl http://localhost:5000/api/status

# Debug information
curl http://localhost:5000/api/debug

# Test optimization
curl -X POST http://localhost:5000/api/optimize \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello world"}'
```

## 📚 API Reference

### Model Classes

- `A2CAgent`: Main A2C agent implementation
- `Actor`: Policy network for action selection
- `Critic`: Value network for state evaluation
- `A2CBuffer`: Experience replay buffer
- `PromptEnvironment`: RL environment for prompt optimization

### Key Methods

- `select_action(state)`: Choose action given state
- `train_step(...)`: Update networks with experience
- `save(path)`: Save trained model
- `load(path)`: Load pre-trained model

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is part of an academic assignment for IoT and Reinforcement Learning.

## 🙏 Acknowledgments

- Groq for providing LLM APIs
- Sentence Transformers for text embeddings
- PyTorch for deep learning framework
- Flask for web application framework 