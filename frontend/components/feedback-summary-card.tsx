"use client"

import { MessageCircle, ThumbsUp, ThumbsDown, Brain, Sparkles } from "lucide-react"

interface FeedbackSummaryCardProps {
  originalPrompt: string
  aiResponse: string
  manualFeedback?: "up" | "down" | null
  manualFeedbackText?: string
  metaPrompt?: string
  previewMetrics?: {
    clarity?: number
    sentiment?: number
    reward?: number
  }
}

export default function FeedbackSummaryCard({
  originalPrompt,
  aiResponse,
  manualFeedback,
  manualFeedbackText,
  metaPrompt,
  previewMetrics,
}: FeedbackSummaryCardProps) {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4 md:p-6 mb-6">
      <div className="flex items-center gap-2 mb-4">
        <Sparkles className="text-blue-500" size={20} />
        <h2 className="text-lg font-semibold text-gray-800 dark:text-gray-200">Feedback Context</h2>
      </div>

      <div className="space-y-4">
        {/* Original Prompt */}
        <div>
          <div className="flex items-center gap-2 mb-2">
            <MessageCircle size={16} className="text-gray-500 dark:text-gray-400" />
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Original Prompt</span>
          </div>
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
            <p className="text-sm text-gray-900 dark:text-gray-100 break-words">{originalPrompt}</p>
          </div>
        </div>

        {/* AI Response */}
        <div>
          <div className="flex items-center gap-2 mb-2">
            <Brain size={16} className="text-gray-500 dark:text-gray-400" />
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">AI Response</span>
          </div>
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
            <p className="text-sm text-gray-900 dark:text-gray-100 break-words">{aiResponse}</p>
          </div>
        </div>

        {/* Manual Feedback */}
        {manualFeedback && (
          <div>
            <div className="flex items-center gap-2 mb-2">
              {manualFeedback === "up" ? (
                <ThumbsUp size={16} className="text-green-500" />
              ) : (
                <ThumbsDown size={16} className="text-red-500" />
              )}
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Manual Feedback</span>
            </div>
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
              <div className="flex items-center gap-2 mb-2">
                <span
                  className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${
                    manualFeedback === "up"
                      ? "bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400"
                      : "bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400"
                  }`}
                >
                  {manualFeedback === "up" ? <ThumbsUp size={12} /> : <ThumbsDown size={12} />}
                  {manualFeedback === "up" ? "Positive" : "Negative"}
                </span>
              </div>
              {manualFeedbackText && (
                <p className="text-sm text-gray-900 dark:text-gray-100 break-words">{manualFeedbackText}</p>
              )}
            </div>
          </div>
        )}

        {/* Meta Prompt */}
        {metaPrompt && (
          <div>
            <div className="flex items-center gap-2 mb-2">
              <Sparkles size={16} className="text-purple-500" />
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Meta Prompt</span>
            </div>
            <div className="bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded-lg p-3">
              <p className="text-sm text-purple-900 dark:text-purple-100 break-words">{metaPrompt}</p>
            </div>
          </div>
        )}

        {/* Preview Metrics */}
        {previewMetrics && (
          <div>
            <div className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Preview Metrics</div>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              {previewMetrics.clarity && (
                <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-3 text-center">
                  <div className="text-lg font-semibold text-blue-600 dark:text-blue-400">
                    {previewMetrics.clarity.toFixed(1)}
                  </div>
                  <div className="text-xs text-blue-800 dark:text-blue-300">Clarity</div>
                </div>
              )}
              {previewMetrics.sentiment !== undefined && (
                <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-3 text-center">
                  <div className="text-lg font-semibold text-green-600 dark:text-green-400">
                    {previewMetrics.sentiment.toFixed(2)}
                  </div>
                  <div className="text-xs text-green-800 dark:text-green-300">Sentiment</div>
                </div>
              )}
              {previewMetrics.reward && (
                <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-3 text-center">
                  <div className="text-lg font-semibold text-purple-600 dark:text-purple-400">
                    {previewMetrics.reward.toFixed(3)}
                  </div>
                  <div className="text-xs text-purple-800 dark:text-purple-300">Reward</div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
