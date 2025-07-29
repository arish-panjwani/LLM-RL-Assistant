# PPO (Proximal Policy Optimization) Model

This is a PPO implementation for reinforcement learning-based prompt optimization using Groq APIs with **user feedback support** and **dynamic prompt generation**.

## Features

- **PPO Algorithm**: Implements Proximal Policy Optimization for stable policy updates
- **Prompt Optimization**: Dynamically improves prompts for better LLM responses
- **User Feedback Integration**: Collects and incorporates user satisfaction feedback
- **Dynamic Prompt Templates**: Non-hardcoded prompt optimization strategies
- **Reward Function**: Multi-component reward including:
  - Cosine similarity between original and refined prompts
  - Clarity rating
  - Sentiment analysis
  - Hallucination detection
  - Redundancy penalty
  - **User feedback bonus/penalty**
- **Secure API Management**: Environment variables for API keys with proper git/docker ignore
- **Flexible Deployment**: Multiple modes for training and deployment
- **Web Demo**: Interactive web application for testing

## Files Structure

```
PPO/
├── model.py                    # PPO agent implementation
├── main.py                     # Training script with user feedback
├── deploy.py                   # Production deployment script
├── interactive_inference.py    # Interactive testing with feedback
├── utils.py                    # Environment and reward functions
├── test_ppo.py                # PPO model tests
├── test_env.py                # Environment tests
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore rules
├── .dockerignore              # Docker ignore rules
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Docker compose configuration
├── README.md                  # This file
└── demo/                      # Web application demo
    ├── app.py                 # Flask webapp
    ├── run_demo.py           # Demo startup script
    ├── requirements.txt      # Demo dependencies
    ├── README.md            # Demo documentation
    └── templates/
        └── index.html       # Web interface
```

## Secure Setup

### 1. Create Virtual Environment
```bash
py -m venv ppo_env
ppo_env\Scripts\activate.bat
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up API Keys (Secure Method)
Create a `.env` file with your API keys:
```
GROQ_API_KEY=your_groq_api_key_here
WOLFRAM_APP_ID=your_wolfram_app_id_here
GOOGLE_API_KEY=your_google_api_key_here
```

**Required API Keys:**
- `GROQ_API_KEY`: Your Groq API key
- `WOLFRAM_APP_ID`: Your Wolfram Alpha App ID  
- `GOOGLE_API_KEY`: Your Google API key

## Usage

### Training Options

#### Interactive Mode Selection
```bash
python main.py
```
This will prompt you to choose:
1. **Training mode** (automated training)
2. **Interactive mode** (training with feedback)
3. **Deployment mode** (real-time inference only)

#### Command Line Mode Selection
```bash
# Automated training
python main.py train

# Interactive training with feedback
python main.py interactive

# Deployment mode (no training)
python main.py deploy
```

### Production Deployment
```bash
python deploy.py
```
Loads the trained model and runs in production mode for prompt optimization.

### Interactive Testing
```bash
python interactive_inference.py
```
Test the trained model with your own prompts and provide feedback.

### Web Demo (Recommended for Testing)
```bash
cd demo
pip install -r requirements.txt
python run_demo.py
```
Then open http://localhost:5000 in your browser for a beautiful web interface!

### Testing
```bash
python test_ppo.py
python test_env.py
```

### Docker
```bash
# Build and run with Docker
docker-compose up --build

# Or build manually
docker build -t ppo-model .
docker run --env-file .env ppo-model
```

## Web Demo Features

The demo webapp provides:
- 🎨 **Beautiful UI**: Modern, responsive design
- 🔄 **Real-time Optimization**: Instant prompt optimization
- 📊 **Feedback System**: Click to provide satisfaction feedback
- 📈 **Statistics**: Live feedback statistics and history
- 🤖 **LLM Integration**: See actual LLM responses
- 📱 **Mobile Friendly**: Works on all devices

## Deployment Modes

### 🎯 **Training Mode**
- **Purpose**: Train the PPO model
- **Episodes**: Configurable (default: 10)
- **Feedback**: Optional user feedback
- **Output**: Saves trained model

### 🚀 **Deployment Mode**
- **Purpose**: Production inference
- **Episodes**: 0 (no training)
- **Feedback**: None
- **Output**: Real-time prompt optimization

### 📝 **Interactive Mode**
- **Purpose**: Training with user feedback
- **Episodes**: Configurable
- **Feedback**: Required for each prompt
- **Output**: Trained model + feedback statistics

## User Feedback Features

### ✅ **User Feedback Integration**
- **Real-time feedback**: Users can provide satisfaction feedback during training
- **Feedback storage**: All feedback is stored for analysis and potential retraining
- **Reward modification**: User feedback directly affects the reward function
- **Statistics tracking**: Satisfaction rates and feedback history

### ✅ **Dynamic Prompt Generation**
- **Multiple templates**: 5 different prompt optimization strategies
- **Embedding-based selection**: Template selection based on prompt embedding
- **No hardcoded prompts**: All optimization strategies are dynamic

### Feedback Modes
1. **Automated Mode**: Traditional training without user input
2. **Interactive Mode**: User provides feedback on each optimization
3. **Inference Mode**: Test trained model with custom prompts

## Security Features

- **Environment Variables**: API keys stored in `.env` file
- **Git Ignore**: `.env` files automatically excluded from version control
- **Docker Ignore**: Sensitive files excluded from Docker builds
- **No Hardcoded Keys**: All API keys loaded from environment variables

## Model Architecture

- **Actor Network**: 3-layer MLP with tanh output and learnable standard deviation
- **Critic Network**: 3-layer MLP for value function estimation
- **PPO Features**:
  - GAE (Generalized Advantage Estimation)
  - Clipped surrogate objective
  - Value function clipping
  - Entropy bonus for exploration
  - Gradient clipping

## Reward Components

1. **Cosine Similarity**: Measures semantic similarity between original and refined prompts
2. **Clarity Rating**: Random rating (5-10) for prompt clarity
3. **Sentiment Analysis**: VADER sentiment analysis of LLM responses
4. **Hallucination Detection**: Penalizes uncertain responses
5. **Redundancy Penalty**: Penalizes repetitive words in refined prompts
6. **User Feedback**: +2.0 bonus for satisfied users, -1.0 penalty for dissatisfied users

## API Integration

- **Groq API**: For LLM responses using Llama3-8b-8192 model
- **Wolfram Alpha API**: For computational knowledge queries
- **Google API**: For search functionality
- **Sentence Transformers**: For text embedding using all-MiniLM-L6-v2
- **NLTK VADER**: For sentiment analysis

## Training Process

1. Encode original prompt using sentence transformer
2. Generate refined prompt using dynamic templates
3. Get LLM response for the refined prompt
4. Calculate reward based on multiple factors
5. **Collect user feedback** (if in interactive mode)
6. **Modify reward** based on user satisfaction
7. Update PPO policy using the final reward

## Example Usage

### Training Session
```bash
$ python main.py train
🚀 PPO Training with User Feedback
==================================================
✅ Loaded pre-trained model from saved_model/ppo_actor.pth
🤖 Automated Training Mode
Training will proceed without user feedback.

Enter number of training episodes (default: 10): 5
🎯 Episode 1/5
------------------------------
```

### Deployment Session
```bash
$ python deploy.py
🚀 PPO Model Deployment
========================================
🖥️  Using device: cpu
✅ Loaded trained model successfully

🎯 Model ready for deployment!
Enter prompts to optimize, or 'quit' to exit.

📝 Enter prompt to optimize: How do I cook rice?
🔄 Optimizing: 'How do I cook rice?'
📝 Original: How do I cook rice?
🔄 Optimized: Make this prompt clearer and more detailed: How do I cook rice?
--------------------------------------------------
```

### Web Demo Session
```bash
$ cd demo
$ python run_demo.py
🚀 PPO Demo WebApp Startup
========================================
✅ All dependencies are installed!
✅ .env file found

🌐 Starting Flask webapp...
📱 Open your browser and go to: http://localhost:5000
```

This enhanced model now supports flexible deployment modes, command-line options, and a beautiful web demo! 🎉 