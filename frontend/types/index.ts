export type RLModel = "PPO" | "DDPG" | "A2C" | "SAC"

export interface Message {
  id: string
  content: string
  sender: "user" | "assistant"
  timestamp: Date
  feedback?: "up" | "down" | null
  originalPrompt?: string
  modifiedPrompt?: string
  feedbackSummary?: FeedbackSummary
  finalReward?: number
  imageUrl?: string
  imageFile?: File
}

export interface FeedbackSummary {
  promptClarity: number
  responseConsistency: number
  lexicalDiversity: number
  sentimentScore: number
  hallucinationFlag: boolean
  factAccuracy: number
}

export interface FeedbackStep {
  id: string
  name: string
  description: string
  status: "pending" | "loading" | "completed" | "error"
  result?: number | boolean
  duration?: number
}

export interface PromptLog {
  id: string
  originalPrompt: string
  modifiedPrompt: string
  aiResponse: string
  feedbackSummary: FeedbackSummary
  finalReward: number
  timestamp: Date
  model: RLModel
}
