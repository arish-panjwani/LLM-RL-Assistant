# LLM-RL Assistant

A sophisticated conversational AI assistant built with Next.js that integrates reinforcement learning models for enhanced response quality. The application features real-time feedback evaluation, multimedia messaging, and comprehensive analytics.

## 🚀 Features

### Core Functionality
- **Multi-Modal Chat Interface**: Text, voice input, and image support
- **Reinforcement Learning Integration**: PPO, DDPG, A2C, and SAC models
- **Real-Time Feedback Pipeline**: Live evaluation of AI responses
- **Voice Recognition & TTS**: Speech-to-text and text-to-speech capabilities
- **Camera Integration**: Take photos or upload images with optional text
- **Dark Mode Support**: Seamless theme switching
- **Responsive Design**: Mobile-first approach with touch-friendly interactions

### Advanced Features
- **Live Feedback Flow**: Step-by-step visualization of AI evaluation metrics
- **Prompt Logs**: Historical tracking of prompts, responses, and feedback
- **Manual Feedback System**: User rating and feedback collection
- **Audio Feedback**: Soft tick sounds for pipeline steps
- **Celebratory Rewards**: Visual feedback for model improvements

## 📋 Prerequisites

- Node.js 18+ 
- npm or yarn
- Modern web browser with camera/microphone support
- Backend server (Flask/FastAPI recommended)

## 🛠️ Installation

### Frontend Setup

1. **Clone the repository**
   \`\`\`bash
   git clone <repository-url>
   cd llm-rl-assistant
   \`\`\`

2. **Install dependencies**
   \`\`\`bash
   npm install
   # or
   yarn install
   \`\`\`

3. **Configure environment variables**
   Create a \`.env.local\` file:
   \`\`\`env
   # API Configuration
   NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
   NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
   
   # Feature Flags
   NEXT_PUBLIC_USE_MOCK_API=false
   NEXT_PUBLIC_ENABLE_VOICE=true
   NEXT_PUBLIC_ENABLE_CAMERA=true
   
   # External APIs (if using)
   GROQ_API_KEY=your_groq_api_key
   OPENAI_API_KEY=your_openai_api_key
   \`\`\`

4. **Start development server**
   \`\`\`bash
   npm run dev
   # or
   yarn dev
   \`\`\`

## 🔧 Backend Integration

### API Endpoints Required

The frontend expects the following REST API endpoints:

#### 1. Chat API
\`\`\`
POST /api/chat
Content-Type: application/json

Request:
{
  "message": "string",
  "model": "PPO" | "DDPG" | "A2C" | "SAC",
  "image": "base64_string" (optional)
}

Response:
{
  "response": "string",
  "modifiedPrompt": "string",
  "messageId": "string"
}
\`\`\`

#### 2. Feedback API
\`\`\`
POST /api/feedback
Content-Type: application/json

Request:
{
  "messageId": "string",
  "feedback": "up" | "down",
  "model": "PPO" | "DDPG" | "A2C" | "SAC",
  "text": "string" (optional),
  "rating": number (optional)
}

Response:
{
  "success": boolean,
  "message": "string"
}
\`\`\`

#### 3. Feedback Evaluation API
\`\`\`
POST /api/feedback-evaluation
Content-Type: application/json

Request:
{
  "messageId": "string",
  "content": "string"
}

Response:
{
  "promptClarity": number,
  "responseConsistency": number,
  "lexicalDiversity": number,
  "sentimentScore": number,
  "hallucinationFlag": boolean,
  "factAccuracy": number
}
\`\`\`

#### 4. Prompt Processing API
\`\`\`
POST /api/process-prompt
Content-Type: application/json

Request:
{
  "prompt": "string",
  "model": "PPO" | "DDPG" | "A2C" | "SAC"
}

Response:
{
  "modifiedPrompt": "string"
}
\`\`\`

#### 5. Hallucination Check API
\`\`\`
POST /api/hallucination-check
Content-Type: application/json

Request:
{
  "content": "string"
}

Response:
{
  "isHallucination": boolean,
  "confidence": number
}
\`\`\`

### WebSocket Integration

#### Connection Setup
\`\`\`javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = () => {
  console.log('WebSocket connected');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  handleRealtimeMessage(data);
};
\`\`\`

#### Message Types
\`\`\`typescript
// Real-time response streaming
{
  "type": "response_chunk",
  "messageId": "string",
  "chunk": "string",
  "isComplete": boolean
}

// Feedback step updates
{
  "type": "feedback_step",
  "stepId": "string",
  "status": "loading" | "completed" | "error",
  "result": any
}

// Model training updates
{
  "type": "model_update",
  "model": "PPO" | "DDPG" | "A2C" | "SAC",
  "reward": number,
  "metrics": object
}
\`\`\`

## 🐍 Flask Backend Example

### Basic Flask Setup

\`\`\`python
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app)

# Your RL models
models = {
    'PPO': None,  # Initialize your PPO model
    'DDPG': None,  # Initialize your DDPG model
    'A2C': None,   # Initialize your A2C model
    'SAC': None    # Initialize your SAC model
}

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message')
    model_name = data.get('model', 'PPO')
    image = data.get('image')  # base64 encoded image
    
    # Process with your RL model
    model = models[model_name]
    
    # Generate response
    response = generate_response(model, message, image)
    modified_prompt = process_prompt(model, message)
    
    return jsonify({
        'response': response,
        'modifiedPrompt': modified_prompt,
        'messageId': generate_message_id()
    })

@app.route('/api/feedback', methods=['POST'])
def feedback():
    data = request.json
    message_id = data.get('messageId')
    feedback = data.get('feedback')  # 'up' or 'down'
    model_name = data.get('model')
    
    # Update model with feedback
    update_model_with_feedback(models[model_name], message_id, feedback)
    
    return jsonify({'success': True, 'message': 'Feedback recorded'})

@app.route('/api/feedback-evaluation', methods=['POST'])
def feedback_evaluation():
    data = request.json
    content = data.get('content')
    
    # Evaluate content
    metrics = evaluate_content(content)
    
    return jsonify(metrics)

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connected', {'status': 'Connected to LLM-RL Assistant'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app, debug=True, port=8000)
\`\`\`

### Advanced Integration Functions

\`\`\`python
import asyncio
from typing import Dict, Any
import numpy as np

def generate_response(model, message: str, image: str = None) -> str:
    """Generate AI response using RL model"""
    # Preprocess input
    processed_input = preprocess_input(message, image)
    
    # Get model prediction
    response = model.predict(processed_input)
    
    # Post-process response
    return postprocess_response(response)

def process_prompt(model, prompt: str) -> str:
    """Process prompt with RL model optimization"""
    # Apply model-specific prompt engineering
    if model.name == 'PPO':
        return f"[PPO optimized] {prompt}"
    elif model.name == 'DDPG':
        return f"[DDPG enhanced] {prompt}"
    # ... other models
    
    return prompt

def evaluate_content(content: str) -> Dict[str, Any]:
    """Evaluate content for various metrics"""
    return {
        'promptClarity': calculate_clarity(content),
        'responseConsistency': calculate_consistency(content),
        'lexicalDiversity': calculate_diversity(content),
        'sentimentScore': analyze_sentiment(content),
        'hallucinationFlag': detect_hallucination(content),
        'factAccuracy': verify_facts(content)
    }

def update_model_with_feedback(model, message_id: str, feedback: str):
    """Update RL model with user feedback"""
    reward = 1.0 if feedback == 'up' else -1.0
    
    # Update model parameters
    model.update_with_reward(message_id, reward)
    
    # Emit real-time update
    socketio.emit('model_update', {
        'model': model.name,
        'reward': reward,
        'metrics': model.get_metrics()
    })

async def stream_response(model, message: str):
    """Stream response in real-time"""
    response_chunks = model.generate_streaming(message)
    
    for chunk in response_chunks:
        socketio.emit('response_chunk', {
            'messageId': 'current_message_id',
            'chunk': chunk,
            'isComplete': False
        })
        await asyncio.sleep(0.1)  # Small delay for streaming effect
    
    # Mark as complete
    socketio.emit('response_chunk', {
        'messageId': 'current_message_id',
        'chunk': '',
        'isComplete': True
    })
\`\`\`

## 🔄 Configuration

### Switching Between Mock and Real APIs

Update \`config/urls.ts\`:

\`\`\`typescript
// For development with mock APIs
export const USE_MOCK_API = true

// For production with real backend
export const USE_MOCK_API = false

export const API_URLS = {
  CHAT_API: "http://your-backend.com/api/chat",
  WEBSOCKET_URL: "ws://your-backend.com/ws",
  // ... other URLs
}
\`\`\`

### Model Configuration

Update model settings in \`mock/mock-data.ts\` or your backend:

\`\`\`typescript
const models = [
  { value: "PPO", label: "PPO", description: "Proximal Policy Optimization" },
  { value: "DDPG", label: "DDPG", description: "Deep Deterministic Policy Gradient" },
  { value: "A2C", label: "A2C", description: "Advantage Actor-Critic" },
  { value: "SAC", label: "SAC", description: "Soft Actor-Critic" },
]
\`\`\`

## 🚀 Deployment

### Frontend Deployment (Vercel)

1. **Connect to Vercel**
   \`\`\`bash
   npm install -g vercel
   vercel login
   vercel
   \`\`\`

2. **Environment Variables**
   Set in Vercel dashboard:
   - \`NEXT_PUBLIC_API_BASE_URL\`
   - \`NEXT_PUBLIC_WS_URL\`
   - \`NEXT_PUBLIC_USE_MOCK_API=false\`

### Backend Deployment

#### Docker Setup
\`\`\`dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "app.py"]
\`\`\`

#### Docker Compose
\`\`\`yaml
version: '3.8'
services:
  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - FLASK_ENV=production
      - DATABASE_URL=postgresql://...
    volumes:
      - ./models:/app/models
  
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
\`\`\`

## 🧪 Testing

### Frontend Testing
\`\`\`bash
# Unit tests
npm run test

# E2E tests
npm run test:e2e

# Component testing
npm run test:components
\`\`\`

### Backend Testing
\`\`\`python
# Test API endpoints
python -m pytest tests/test_api.py

# Test WebSocket connections
python -m pytest tests/test_websocket.py

# Test RL models
python -m pytest tests/test_models.py
\`\`\`

## 📊 Monitoring & Analytics

### Frontend Metrics
- User interaction tracking
- Response time monitoring
- Error rate analysis
- Feature usage statistics

### Backend Metrics
- Model performance tracking
- API response times
- WebSocket connection health
- Resource utilization

## 🔒 Security Considerations

1. **API Authentication**: Implement JWT or API key authentication
2. **Rate Limiting**: Prevent API abuse
3. **Input Validation**: Sanitize all user inputs
4. **CORS Configuration**: Restrict origins in production
5. **WebSocket Security**: Implement connection authentication

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **WebSocket Connection Failed**
   - Check backend server is running
   - Verify WebSocket URL in configuration
   - Check firewall settings

2. **Speech Recognition Not Working**
   - Ensure HTTPS in production
   - Check microphone permissions
   - Verify browser compatibility

3. **Camera Access Denied**
   - Check browser permissions
   - Ensure HTTPS for camera access
   - Verify device camera availability

4. **Model Loading Issues**
   - Check model file paths
   - Verify model format compatibility
   - Ensure sufficient memory

### Debug Mode

Enable debug logging:
\`\`\`typescript
// In config/urls.ts
export const DEBUG_MODE = true
\`\`\`

### Performance Optimization

1. **Frontend**
   - Enable React strict mode
   - Implement code splitting
   - Optimize image loading
   - Use service workers

2. **Backend**
   - Implement caching
   - Use connection pooling
   - Optimize model inference
   - Monitor memory usage

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review troubleshooting guide
- Contact the development team

---

**Happy coding! 🚀**
\`\`\`

The layout issues have been fixed by:
1. Removing extra height constraints that were causing spacing issues
2. Properly structuring the chat window with flexbox
3. Ensuring the input area is properly aligned and contained
4. Removing unnecessary padding that was causing misalignment

The README provides comprehensive instructions for backend integration with all necessary APIs, WebSocket setup, and deployment guidelines.
