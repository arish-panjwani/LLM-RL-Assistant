# API Setup Guide

## Problem
The API testing is showing HTTP 500 errors because the `GROQ_API_KEY` environment variable is not set.

## Solution

### Option 1: Set Environment Variable (Recommended)

1. **Get a Groq API Key:**
   - Visit https://console.groq.com/
   - Sign up and get your API key

2. **Set the Environment Variable:**

   **Windows (PowerShell):**
   ```powershell
   $env:GROQ_API_KEY="your_actual_api_key_here"
   ```

   **Windows (Command Prompt):**
   ```cmd
   set GROQ_API_KEY=your_actual_api_key_here
   ```

   **Linux/Mac:**
   ```bash
   export GROQ_API_KEY=your_actual_api_key_here
   ```

3. **Restart the Docker container:**
   ```bash
   docker-compose down
   docker-compose up --build
   ```

### Option 2: Create a .env file

1. Create a `.env` file in the project root:
   ```
   GROQ_API_KEY=your_actual_api_key_here
   ```

2. Restart the Docker container:
   ```bash
   docker-compose down
   docker-compose up --build
   ```

### Option 3: Use Fallback Mode (No API Key Required)

The system has been updated to work without an API key using fallback responses. The API will still function but with reduced capabilities.

## Current Status

✅ **Health Endpoint:** Working  
✅ **Model Info Endpoint:** Working  
⚠️ **Optimization Endpoint:** Using fallback mode (no API key)

## Testing

After setting up the API key, run the test again:

```bash
python demo/api_testing.py
```

You should see successful responses instead of 500 errors.

## Fallback Mode Features

When no API key is provided, the system:
- ✅ Responds to all requests
- ✅ Provides context-aware responses
- ✅ Includes basic metrics
- ⚠️ Uses predefined patterns instead of LLM-generated content
- ⚠️ Has reduced response quality and diversity 