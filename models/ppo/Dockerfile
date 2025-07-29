# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy local files to container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK lexicon at build time
RUN python -c "import nltk; nltk.download('vader_lexicon')"

# Set environment variables 
ENV GROQ_API_BASE=https://api.groq.com/openai/v1

# Run inference by default
CMD ["python", "inference_test.py"] 