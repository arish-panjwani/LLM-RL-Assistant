"use client"

import { useState, useEffect } from "react"
import { Play, CheckCircle, Loader, AlertCircle, Trophy, Sparkles } from "lucide-react"
import { getMockFeedbackSteps, mockFeedbackEvaluation } from "@/mock/mock-data"
import { AudioHelper } from "@/utils/audio-helper"
import type { FeedbackStep, FeedbackSummary, RLModel } from "@/types"
import FeedbackSummaryCard from "@/components/feedback-summary-card"

export default function LiveFeedbackFlow() {
  const [steps, setSteps] = useState<FeedbackStep[]>(getMockFeedbackSteps())
  const [isRunning, setIsRunning] = useState(false)
  const [currentStep, setCurrentStep] = useState(-1)
  const [results, setResults] = useState<FeedbackSummary | null>(null)
  const [finalReward, setFinalReward] = useState<number | null>(null)
  const [showCelebration, setShowCelebration] = useState(false)
  const [selectedModel, setSelectedModel] = useState<RLModel>("PPO")
  const [feedbackContext, setFeedbackContext] = useState<{
    originalPrompt: string
    aiResponse: string
    manualFeedback?: "up" | "down"
    manualFeedbackText?: string
    metaPrompt?: string
  } | null>(null)

  // Initialize audio helper
  useEffect(() => {
    AudioHelper.initializeTickSound()
  }, [])

  // Get selected model from global state
  useEffect(() => {
    if ((window as any).getSelectedModel) {
      setSelectedModel((window as any).getSelectedModel())
    }
  }, [])

  const playTickSound = () => {
    AudioHelper.playTickSound()
  }

  const startFeedbackPipeline = async () => {
    setIsRunning(true)
    setCurrentStep(0)
    setResults(null)
    setFinalReward(null)
    setShowCelebration(false)

    // Reset all steps
    setSteps(getMockFeedbackSteps())

    // Simulate step-by-step evaluation
    for (let i = 0; i < steps.length - 1; i++) {
      setCurrentStep(i)

      // Update current step to loading
      setSteps((prev) => prev.map((step, index) => (index === i ? { ...step, status: "loading" } : step)))

      // Simulate processing time
      await new Promise((resolve) => setTimeout(resolve, 1000 + Math.random() * 1000))

      // Update current step to completed and play tick sound
      setSteps((prev) => prev.map((step, index) => (index === i ? { ...step, status: "completed" } : step)))
      playTickSound()
    }

    // Final step - compute reward
    setCurrentStep(steps.length - 1)
    setSteps((prev) => prev.map((step, index) => (index === steps.length - 1 ? { ...step, status: "loading" } : step)))

    // Get mock results
    const mockResults = await mockFeedbackEvaluation("test", "test content")
    setResults(mockResults)

    // Calculate final reward
    const reward =
      (mockResults.promptClarity / 10) * 0.2 +
      mockResults.responseConsistency * 0.2 +
      mockResults.lexicalDiversity * 0.15 +
      ((mockResults.sentimentScore + 1) / 2) * 0.1 +
      (mockResults.hallucinationFlag ? 0 : 0.2) +
      mockResults.factAccuracy * 0.15

    setFinalReward(reward)

    // Complete final step
    await new Promise((resolve) => setTimeout(resolve, 1000))
    setSteps((prev) =>
      prev.map((step, index) => (index === steps.length - 1 ? { ...step, status: "completed" } : step)),
    )
    playTickSound()

    // Show celebration after a brief delay
    setTimeout(() => {
      setShowCelebration(true)
    }, 500)

    setIsRunning(false)
    setCurrentStep(-1)
  }

  const getStepIcon = (step: FeedbackStep, index: number) => {
    if (step.status === "loading") {
      return <Loader className="animate-spin" size={20} />
    } else if (step.status === "completed") {
      return <CheckCircle className="text-green-500" size={20} />
    } else if (step.status === "error") {
      return <AlertCircle className="text-red-500" size={20} />
    } else {
      return <div className="w-5 h-5 rounded-full border-2 border-gray-300 dark:border-gray-600" />
    }
  }

  const getMetricValue = (key: keyof FeedbackSummary) => {
    if (!results) return "—"
    const value = results[key]
    if (typeof value === "boolean") return value ? "True" : "False"
    if (typeof value === "number") return value.toFixed(3)
    return "—"
  }

  const setFeedbackContextData = (data: {
    originalPrompt: string
    aiResponse: string
    manualFeedback?: "up" | "down"
    manualFeedbackText?: string
  }) => {
    const metaPrompt = `Original: "${data.originalPrompt}" | Response: "${data.aiResponse}" | User Feedback: ${data.manualFeedback === "up" ? "Positive" : "Negative"}${data.manualFeedbackText ? ` - "${data.manualFeedbackText}"` : ""}`

    setFeedbackContext({
      ...data,
      metaPrompt,
    })
  }

  // Expose this method for external use
  useEffect(() => {
    ;(window as any).setLiveFeedbackContext = setFeedbackContextData
    ;(window as any).getSelectedModel = () => selectedModel
  }, [selectedModel])

  return (
    <div className="h-full overflow-y-auto p-4 md:p-6">
      <div className="mb-6 md:mb-8">
        <h1 className="text-xl md:text-2xl font-bold text-gray-800 dark:text-gray-200 mb-2">Live Feedback Flow</h1>
        <p className="text-sm md:text-base text-gray-600 dark:text-gray-400">
          Real-time visualization of AI feedback evaluation pipeline
        </p>
      </div>

      {/* Feedback Summary Card */}
      {feedbackContext && (
        <FeedbackSummaryCard
          originalPrompt={feedbackContext.originalPrompt}
          aiResponse={feedbackContext.aiResponse}
          manualFeedback={feedbackContext.manualFeedback}
          manualFeedbackText={feedbackContext.manualFeedbackText}
          metaPrompt={feedbackContext.metaPrompt}
          previewMetrics={
            results
              ? {
                  clarity: results.promptClarity,
                  sentiment: results.sentimentScore,
                  reward: finalReward || undefined,
                }
              : undefined
          }
        />
      )}

      {/* Start Button */}
      <div className="mb-6 md:mb-8">
        <button
          onClick={startFeedbackPipeline}
          disabled={isRunning}
          className="w-full sm:w-auto flex items-center justify-center gap-2 px-4 md:px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-sm md:text-base touch-manipulation"
        >
          <Play size={18} />
          {isRunning ? "Running Pipeline..." : "Start Feedback Evaluation"}
        </button>
      </div>

      {/* Pipeline Steps - Mobile First */}
      <div className="space-y-3 md:space-y-4 mb-6 md:mb-8">
        {steps.map((step, index) => (
          <div
            key={step.id}
            className={`p-3 md:p-4 rounded-lg border transition-all ${
              step.status === "loading"
                ? "border-blue-300 bg-blue-50 dark:bg-blue-900/20 dark:border-blue-700"
                : step.status === "completed"
                  ? "border-green-300 bg-green-50 dark:bg-green-900/20 dark:border-green-700"
                  : step.status === "error"
                    ? "border-red-300 bg-red-50 dark:bg-red-900/20 dark:border-red-700"
                    : "border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800"
            }`}
          >
            <div className="flex items-start gap-3">
              <div className="flex-shrink-0 mt-0.5">{getStepIcon(step, index)}</div>
              <div className="flex-1 min-w-0">
                <h3 className="font-medium text-gray-800 dark:text-gray-200 text-sm md:text-base break-words">
                  {step.name}
                </h3>
                <p className="text-xs md:text-sm text-gray-600 dark:text-gray-400 mt-1 break-words">
                  {step.description}
                </p>
              </div>
              {step.status === "loading" && (
                <div className="flex-shrink-0 text-xs md:text-sm text-blue-600 dark:text-blue-400 font-medium">
                  Processing...
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Results Summary - Mobile First */}
      {results && (
        <div className="space-y-4 md:space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4 md:p-6">
            <h2 className="text-lg md:text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
              <Trophy className="text-yellow-500" size={20} />
              Evaluation Results
            </h2>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 md:gap-4 mb-4 md:mb-6">
              {/* Metric cards with responsive text */}
              <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                <div className="text-xs md:text-sm text-gray-600 dark:text-gray-400">Prompt Clarity</div>
                <div className="text-base md:text-lg font-semibold text-gray-800 dark:text-gray-200 break-words">
                  {getMetricValue("promptClarity")}/10
                </div>
              </div>

              <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                <div className="text-xs md:text-sm text-gray-600 dark:text-gray-400">Response Consistency</div>
                <div className="text-base md:text-lg font-semibold text-gray-800 dark:text-gray-200 break-words">
                  {getMetricValue("responseConsistency")}
                </div>
              </div>

              <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                <div className="text-xs md:text-sm text-gray-600 dark:text-gray-400">Lexical Diversity</div>
                <div className="text-base md:text-lg font-semibold text-gray-800 dark:text-gray-200 break-words">
                  {getMetricValue("lexicalDiversity")}
                </div>
              </div>

              <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                <div className="text-xs md:text-sm text-gray-600 dark:text-gray-400">Sentiment Score</div>
                <div className="text-base md:text-lg font-semibold text-gray-800 dark:text-gray-200 break-words">
                  {getMetricValue("sentimentScore")}
                </div>
              </div>

              <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                <div className="text-xs md:text-sm text-gray-600 dark:text-gray-400">Hallucination Flag</div>
                <div
                  className={`text-base md:text-lg font-semibold ${
                    results.hallucinationFlag ? "text-red-600 dark:text-red-400" : "text-green-600 dark:text-green-400"
                  } break-words`}
                >
                  {getMetricValue("hallucinationFlag")}
                </div>
              </div>

              <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                <div className="text-xs md:text-sm text-gray-600 dark:text-gray-400">Fact Accuracy</div>
                <div className="text-base md:text-lg font-semibold text-gray-800 dark:text-gray-200 break-words">
                  {getMetricValue("factAccuracy")}
                </div>
              </div>
            </div>

            {/* Final Reward - Mobile Optimized */}
            {finalReward !== null && (
              <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
                <div className="text-center">
                  <div className="text-xs md:text-sm text-gray-600 dark:text-gray-400 mb-1">Final Reward Score</div>
                  <div className="text-2xl md:text-3xl font-bold text-blue-600 dark:text-blue-400 break-words">
                    {finalReward.toFixed(3)}
                  </div>
                  <div className="text-xs md:text-sm text-gray-500 dark:text-gray-500">
                    Weighted average of all metrics
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Celebratory Reward Card */}
          {showCelebration && finalReward !== null && (
            <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg border-2 border-purple-200 dark:border-purple-700 p-4 md:p-6 animate-pulse">
              <div className="text-center">
                <div className="flex items-center justify-center gap-2 mb-3">
                  <Sparkles className="text-purple-500 animate-bounce" size={24} />
                  <span className="text-2xl">🎉</span>
                  <Sparkles className="text-pink-500 animate-bounce" size={24} />
                </div>

                <h3 className="text-lg md:text-xl font-bold text-purple-800 dark:text-purple-200 mb-2">
                  Reward Delivered!
                </h3>

                <p className="text-sm md:text-base text-purple-700 dark:text-purple-300 mb-3">
                  Reward of{" "}
                  <span className="font-bold text-purple-900 dark:text-purple-100">{finalReward.toFixed(3)}</span> has
                  been given to the{" "}
                  <span className="font-bold bg-purple-200 dark:bg-purple-800 px-2 py-1 rounded text-purple-900 dark:text-purple-100">
                    {selectedModel}
                  </span>{" "}
                  model
                </p>

                <div className="flex items-center justify-center gap-2 text-sm md:text-base text-purple-600 dark:text-purple-400">
                  <span>Your model just got smarter!</span>
                  <span className="text-lg">🚀</span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
