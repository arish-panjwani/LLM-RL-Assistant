import type { RLModel, FeedbackSummary, FeedbackStep } from "@/types"

// Mock chat response
export function mockChatResponse(message: string, model: RLModel): { response: string; modifiedPrompt: string } {
  const responses = [
    `I understand your question about "${message}". Using the ${model} model, I can provide you with a comprehensive answer. This is a mock response that simulates real AI interaction.`,
    `That's an interesting point about "${message}". The ${model} reinforcement learning model helps me provide more contextually appropriate responses. Here's what I think...`,
    `Thank you for asking about "${message}". With ${model} optimization, I can better understand your intent and provide more helpful responses.`,
  ]

  const modifiedPrompts = [
    `[${model} Enhanced] ${message} - optimized for clarity and context`,
    `[${model} Processed] ${message} - refined for better understanding`,
    `[${model} Optimized] ${message} - enhanced with contextual awareness`,
  ]

  return {
    response: responses[Math.floor(Math.random() * responses.length)],
    modifiedPrompt: modifiedPrompts[Math.floor(Math.random() * modifiedPrompts.length)],
  }
}

// Mock prompt processing
export function mockPromptProcessing(originalPrompt: string, model: RLModel): string {
  const enhancements = {
    PPO: "policy-optimized",
    DDPG: "deterministically-enhanced",
    A2C: "actor-critic-refined",
    SAC: "soft-actor-optimized",
  }

  return `[${model} ${enhancements[model]}] ${originalPrompt}`
}

// Mock feedback evaluation with live steps
export async function mockFeedbackEvaluation(messageId: string, content: string): Promise<FeedbackSummary> {
  // Simulate step-by-step evaluation
  const steps: FeedbackStep[] = [
    { id: "1", name: "Prompt Clarity", description: "Evaluating prompt clarity and structure", status: "pending" },
    { id: "2", name: "Response Consistency", description: "Checking response consistency", status: "pending" },
    { id: "3", name: "Lexical Diversity", description: "Measuring lexical diversity", status: "pending" },
    { id: "4", name: "Sentiment Analysis", description: "Performing sentiment analysis", status: "pending" },
    { id: "5", name: "Hallucination Detection", description: "Detecting potential hallucinations", status: "pending" },
    { id: "6", name: "Fact Verification", description: "Verifying factual accuracy", status: "pending" },
    { id: "7", name: "Final Reward", description: "Computing final reward score", status: "pending" },
  ]

  // Generate mock results
  return {
    promptClarity: Math.random() * 3 + 7, // 7-10 range
    responseConsistency: Math.random() * 0.3 + 0.7, // 0.7-1.0 range
    lexicalDiversity: Math.random() * 0.4 + 0.6, // 0.6-1.0 range
    sentimentScore: (Math.random() - 0.5) * 1.6, // -0.8 to 0.8 range
    hallucinationFlag: Math.random() > 0.85, // 15% chance
    factAccuracy: Math.random() * 0.2 + 0.8, // 0.8-1.0 range
  }
}

// Mock feedback steps for live visualization
export function getMockFeedbackSteps(): FeedbackStep[] {
  return [
    {
      id: "clarity",
      name: "Evaluating Prompt Clarity",
      description: "Analyzing prompt structure and clarity",
      status: "pending",
    },
    {
      id: "consistency",
      name: "Checking Response Consistency",
      description: "Measuring response coherence",
      status: "pending",
    },
    {
      id: "diversity",
      name: "Measuring Lexical Diversity",
      description: "Calculating vocabulary richness",
      status: "pending",
    },
    {
      id: "sentiment",
      name: "Performing Sentiment Analysis",
      description: "Analyzing emotional tone",
      status: "pending",
    },
    {
      id: "hallucination",
      name: "Detecting Hallucination",
      description: "Checking for fabricated information",
      status: "pending",
    },
    {
      id: "accuracy",
      name: "Verifying Factual Accuracy",
      description: "Validating factual claims",
      status: "pending",
    },
    {
      id: "reward",
      name: "Computing Final Reward",
      description: "Calculating overall performance score",
      status: "pending",
    },
  ]
}

// Mock prompt logs data
export function getMockPromptLogs() {
  return [
    {
      id: "1",
      originalPrompt: "What is machine learning?",
      modifiedPrompt: "[PPO policy-optimized] What is machine learning?",
      aiResponse:
        "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
      feedbackSummary: {
        promptClarity: 8.5,
        responseConsistency: 0.92,
        lexicalDiversity: 0.78,
        sentimentScore: 0.1,
        hallucinationFlag: false,
        factAccuracy: 0.95,
      },
      finalReward: 0.87,
      timestamp: new Date(Date.now() - 3600000),
      model: "PPO" as RLModel,
    },
    {
      id: "2",
      originalPrompt: "Explain quantum computing",
      modifiedPrompt: "[DDPG deterministically-enhanced] Explain quantum computing",
      aiResponse:
        "Quantum computing leverages quantum mechanical phenomena like superposition and entanglement to process information in ways that classical computers cannot.",
      feedbackSummary: {
        promptClarity: 7.8,
        responseConsistency: 0.88,
        lexicalDiversity: 0.85,
        sentimentScore: 0.05,
        hallucinationFlag: false,
        factAccuracy: 0.91,
      },
      finalReward: 0.83,
      timestamp: new Date(Date.now() - 7200000),
      model: "DDPG" as RLModel,
    },
    {
      id: "3",
      originalPrompt: "How does neural network training work?",
      modifiedPrompt: "[A2C actor-critic-refined] How does neural network training work?",
      aiResponse:
        "Neural network training involves adjusting weights and biases through backpropagation to minimize loss functions and improve prediction accuracy.",
      feedbackSummary: {
        promptClarity: 9.2,
        responseConsistency: 0.94,
        lexicalDiversity: 0.72,
        sentimentScore: 0.15,
        hallucinationFlag: false,
        factAccuracy: 0.97,
      },
      finalReward: 0.91,
      timestamp: new Date(Date.now() - 10800000),
      model: "A2C" as RLModel,
    },
    {
      id: "4",
      originalPrompt: "What are the applications of reinforcement learning?",
      modifiedPrompt: "[SAC soft-actor-optimized] What are the applications of reinforcement learning?",
      aiResponse:
        "Reinforcement learning has applications in robotics, game playing, autonomous vehicles, recommendation systems, and financial trading algorithms.",
      feedbackSummary: {
        promptClarity: 8.9,
        responseConsistency: 0.89,
        lexicalDiversity: 0.81,
        sentimentScore: 0.2,
        hallucinationFlag: false,
        factAccuracy: 0.93,
      },
      finalReward: 0.89,
      timestamp: new Date(Date.now() - 14400000),
      model: "SAC" as RLModel,
    },
    {
      id: "5",
      originalPrompt: "Describe deep learning architectures",
      modifiedPrompt: "[PPO policy-optimized] Describe deep learning architectures",
      aiResponse:
        "Deep learning architectures include convolutional neural networks for image processing, recurrent networks for sequences, and transformers for attention-based tasks.",
      feedbackSummary: {
        promptClarity: 7.6,
        responseConsistency: 0.86,
        lexicalDiversity: 0.79,
        sentimentScore: 0.08,
        hallucinationFlag: false,
        factAccuracy: 0.89,
      },
      finalReward: 0.81,
      timestamp: new Date(Date.now() - 18000000),
      model: "PPO" as RLModel,
    },
  ]
}
