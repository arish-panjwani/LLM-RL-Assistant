This folder contains the API and WebSocket server running on **Raspberry Pi**.

## Responsibilities
- Accept text, audio, or image input from frontend
- Route prompts to the appropriate RL model
- Forward optimized prompts to Groq API
- Send responses back to the smartphone
- Process image inputs (OCR, tagging, captioning)
- Log all feedback and metrics
- Train RL models using reward signals

## Files
- `main.py` – Entry point (Flask or FastAPI)
- `api_routes.py` – REST endpoints (text/image upload, feedback)
- `websocket_server.py` – Handles real-time communication
- `groq_client.py` – Interfaces with Groq LLM API
- `reward_engine.py` – Combines human + AI feedback into reward signal
- `utils.py` – Shared helpers for formatting, error handling

## Image Support
- Accept image via `/upload` POST endpoint
- Supports JPEG, PNG formats
- Uses OCR (e.g., Tesseract) or vision models to extract text

## To Run
```bash
python main.py
