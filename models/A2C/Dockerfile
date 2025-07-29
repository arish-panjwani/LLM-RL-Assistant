# A2C Prompt Optimization System - Production Ready
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY model_requirements.txt .
RUN pip install --no-cache-dir -r model_requirements.txt

# Copy essential files for deployment
COPY src/ ./src/
COPY config/ ./config/
COPY data/models/ ./data/models/
COPY scripts/ ./scripts/
COPY demo/ ./demo/

# Set environment variables
ENV PYTHONPATH=/app/src
ENV FLASK_APP=demo/app.py
ENV FLASK_ENV=production

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port for Flask app
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5001/health')" || exit 1

# Start the Flask application
CMD ["python", "demo/app.py"] 