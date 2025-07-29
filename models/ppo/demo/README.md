# PPO Demo WebApp

A beautiful, interactive web application for testing the PPO (Proximal Policy Optimization) prompt optimizer.

## Features

- 🎨 **Modern UI**: Beautiful, responsive design with Tailwind CSS
- 🔄 **Real-time Optimization**: Test prompt optimization instantly
- 📊 **Feedback System**: Provide satisfaction feedback on optimizations
- 📈 **Statistics**: Track satisfaction rates and feedback history
- 🤖 **LLM Integration**: See actual LLM responses to optimized prompts
- 📱 **Mobile Friendly**: Works on desktop and mobile devices

## Quick Start

### 1. Install Dependencies
```bash
cd demo
pip install -r requirements.txt
```

### 2. Set Up API Keys
Create a `.env` file in the parent directory with your API keys:
```
GROQ_API_KEY=your_groq_api_key_here
WOLFRAM_APP_ID=your_wolfram_app_id_here
GOOGLE_API_KEY=your_google_api_key_here
```

### 3. Run the Demo
```bash
python run_demo.py
```

### 4. Open Browser
Navigate to: http://localhost:5000

## Usage

### Testing Prompt Optimization
1. Enter a prompt in the text area
2. Click "Optimize Prompt" or press Ctrl+Enter
3. View the original vs optimized prompt
4. See the LLM response to the optimized prompt
5. Provide feedback (Satisfied/Not Satisfied)

### Features
- **Model Status**: Shows if the PPO model is loaded
- **Statistics**: Real-time feedback statistics
- **Recent Feedback**: History of your feedback
- **Responsive Design**: Works on all devices

## File Structure

```
demo/
├── app.py                 # Flask webapp
├── run_demo.py           # Startup script with comprehensive testing
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── templates/
    └── index.html       # Web interface
```

## API Endpoints

- `GET /` - Main page
- `POST /api/optimize` - Optimize a prompt
- `POST /api/feedback` - Submit user feedback
- `GET /api/statistics` - Get feedback statistics
- `GET /api/status` - Get model status

## Troubleshooting

### Model Not Loading
The `run_demo.py` script includes comprehensive testing:
- ✅ Checks all dependencies
- ✅ Verifies model files exist
- ✅ Tests model loading and optimization
- ✅ Validates API keys
- ✅ Only starts webapp if all tests pass

### Common Issues
1. **Missing Dependencies**: Run `pip install -r requirements.txt`
2. **Missing .env File**: Create `.env` in parent directory
3. **Import Errors**: Ensure you're in the demo directory
4. **API Errors**: Check your API keys are valid

### Debug Output
The startup script provides detailed feedback:
```
🚀 PPO Demo WebApp Startup
========================================
✅ All dependencies are installed!
✅ Model files found

🔧 Testing model loading and optimization...
✅ Device: cpu
✅ Encoder loaded
✅ Agent created
✅ Environment created
✅ Optimization pipeline test passed

✅ .env file found
✅ GROQ_API_KEY found
✅ WOLFRAM_APP_ID found
✅ GOOGLE_API_KEY found

✅ All tests passed! Model is ready.
🌐 Starting Flask webapp...
```

## Development

### Adding New Features
1. Modify `app.py` for backend changes
2. Update `templates/index.html` for frontend changes
3. Test with `python run_demo.py`

### Styling
The webapp uses Tailwind CSS for styling. Modify the classes in `index.html` to change the appearance.

## Production Deployment

For production deployment, consider:
- Using a production WSGI server (Gunicorn)
- Setting `debug=False`
- Using environment variables for configuration
- Adding proper error handling and logging

## Screenshots

The webapp features:
- Clean, modern interface
- Real-time prompt optimization
- Interactive feedback system
- Statistics dashboard
- Mobile-responsive design

Enjoy testing your PPO model! 🚀 