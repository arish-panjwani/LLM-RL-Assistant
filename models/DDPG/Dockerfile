# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK lexicon (for sentiment)
RUN python -c "import nltk; nltk.download('vader_lexicon')"

# Set environment variable for Groq API
ENV GROQ_API_BASE=https://api.groq.com/openai/v1

# Expose the Flask API port
EXPOSE 5000

# Start Flask API instead of inference_test
CMD ["python", "app.py"]
