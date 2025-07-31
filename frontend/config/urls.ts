// Centralized API URLs configuration
export const API_URLS = {
  // Groq API
  GROQ_API: "https://api.groq.com/openai/v1/chat/completions",

  // WebSocket for real-time communication
  WEBSOCKET_URL: "wss://your-backend.com/ws",

  // Hallucination Detection API
  HALLUCINATION_CHECK: "https://your-backend.com/api/hallucination-check",

  // Feedback Evaluation Pipeline
  FEEDBACK_EVALUATION: "https://your-backend.com/api/feedback-evaluation",

  // Chat API
  CHAT_API: "https://your-backend.com/api/chat",

  // Prompt Processing
  PROMPT_PROCESSING: "https://your-backend.com/api/process-prompt",

  // TTS API
  TTS_API: "https://your-backend.com/api/tts",

  // Speech Recognition
  SPEECH_RECOGNITION: "https://your-backend.com/api/speech-recognition",
}

// Global flag to switch between mock and real APIs
export const USE_MOCK_API = true

// Mock API delay simulation (in milliseconds)
export const MOCK_API_DELAY = {
  CHAT_RESPONSE: 1500,
  FEEDBACK_STEP: 800,
  PROMPT_PROCESSING: 500,
  HALLUCINATION_CHECK: 1200,
}
