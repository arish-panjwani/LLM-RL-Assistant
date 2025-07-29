# API Setup Guide

## 🚀 Getting Real LLM Responses

To get real LLM responses instead of mock responses, you need to configure the Groq API.

### Step 1: Get Groq API Key

1. Go to [Groq Console](https://console.groq.com/)
2. Sign up or log in
3. Create a new API key
4. Copy the API key

### Step 2: Set Environment Variable

**Option A: Set in terminal (temporary)**
```bash
# Windows PowerShell
$env:GROQ_API_KEY="your-api-key-here"

# Windows Command Prompt
set GROQ_API_KEY=your-api-key-here

# Linux/Mac
export GROQ_API_KEY="your-api-key-here"
```

**Option B: Create .env file (permanent)**
1. Create a file named `.env` in your project root
2. Add this line:
```
GROQ_API_KEY=your-api-key-here
```

### Step 3: Test the Setup

Run the feedback system:
```bash
python scripts/test_feedback.py
```

You should see:
```
✅ Groq API initialized successfully!
✅ Groq API connection test successful!
💬 Real LLM Response: [actual response from Groq]
```

## 🔧 Other APIs (Optional)

### Google API for Fact Verification
1. Get Google API key from [Google Cloud Console](https://console.cloud.google.com/)
2. Set environment variable: `GOOGLE_API_KEY=your-key`
3. Set CSE ID: `GOOGLE_CSE_ID=your-cse-id`

### Wikipedia API
- No setup needed, works automatically

## 🎯 What You'll Get

With Groq API configured:
- ✅ **Real LLM responses** from Groq's models
- ✅ **Better evaluation** of response quality
- ✅ **More accurate learning** from your feedback
- ✅ **Professional-grade** prompt optimization

Without API:
- ⚠️ **Mock responses** for testing
- ⚠️ **Limited evaluation** capabilities
- ⚠️ **Basic learning** from your feedback

## 💡 Tips

- Groq offers free API credits for new users
- The API is very fast (real-time responses)
- No rate limiting issues for testing
- Works with multiple models (llama3-8b, llama3-70b, mixtral, gemma) 