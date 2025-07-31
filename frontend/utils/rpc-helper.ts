import { API_URLS, USE_MOCK_API, MOCK_API_DELAY } from "@/config/urls"
import { mockChatResponse, mockFeedbackEvaluation, mockPromptProcessing } from "@/mock/mock-data"
import type { RLModel, FeedbackSummary } from "@/types"

// Helper function to simulate API delay
const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

export class RPCHelper {
  // Chat API
  static async sendMessage(message: string, model: RLModel): Promise<{ response: string; modifiedPrompt: string }> {
    if (USE_MOCK_API) {
      await delay(MOCK_API_DELAY.CHAT_RESPONSE)
      return mockChatResponse(message, model)
    }

    // TODO: Implement real API call to Flask backend
    try {
      const response = await fetch(API_URLS.CHAT_API, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, model }),
      })
      return await response.json()
    } catch (error) {
      console.error("Chat API error:", error)
      throw error
    }
  }

  // Feedback Evaluation Pipeline
  static async evaluateFeedback(messageId: string, content: string): Promise<FeedbackSummary> {
    if (USE_MOCK_API) {
      return mockFeedbackEvaluation(messageId, content)
    }

    // TODO: Implement real feedback evaluation API
    try {
      const response = await fetch(API_URLS.FEEDBACK_EVALUATION, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messageId, content }),
      })
      return await response.json()
    } catch (error) {
      console.error("Feedback evaluation error:", error)
      throw error
    }
  }

  // Process prompt with RL model
  static async processPrompt(originalPrompt: string, model: RLModel): Promise<string> {
    if (USE_MOCK_API) {
      await delay(MOCK_API_DELAY.PROMPT_PROCESSING)
      return mockPromptProcessing(originalPrompt, model)
    }

    // TODO: Implement real prompt processing API
    try {
      const response = await fetch(API_URLS.PROMPT_PROCESSING, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: originalPrompt, model }),
      })
      const data = await response.json()
      return data.modifiedPrompt
    } catch (error) {
      console.error("Prompt processing error:", error)
      throw error
    }
  }

  // Hallucination check
  static async checkHallucination(content: string): Promise<boolean> {
    if (USE_MOCK_API) {
      await delay(MOCK_API_DELAY.HALLUCINATION_CHECK)
      return Math.random() > 0.8 // 20% chance of hallucination
    }

    // TODO: Implement real hallucination detection API
    try {
      const response = await fetch(API_URLS.HALLUCINATION_CHECK, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content }),
      })
      const data = await response.json()
      return data.isHallucination
    } catch (error) {
      console.error("Hallucination check error:", error)
      throw error
    }
  }

  // WebSocket connection (placeholder)
  static connectWebSocket(): WebSocket | null {
    if (USE_MOCK_API) {
      // TODO: Implement mock WebSocket for development
      console.log("Mock WebSocket connection established")
      return null
    }

    // TODO: Implement real WebSocket connection
    try {
      const ws = new WebSocket(API_URLS.WEBSOCKET_URL)
      return ws
    } catch (error) {
      console.error("WebSocket connection error:", error)
      return null
    }
  }

  // Send feedback to backend
  static async sendFeedback(messageId: string, feedback: "up" | "down", model: RLModel): Promise<void> {
    if (USE_MOCK_API) {
      await delay(300)
      console.log("Mock feedback sent:", { messageId, feedback, model })
      return
    }

    // TODO: Implement real feedback API
    try {
      await fetch(`${API_URLS.CHAT_API}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messageId, feedback, model }),
      })
    } catch (error) {
      console.error("Send feedback error:", error)
      throw error
    }
  }
}
