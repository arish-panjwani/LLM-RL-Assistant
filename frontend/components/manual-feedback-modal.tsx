"use client"

import { useState } from "react"
import { X, ThumbsUp, ThumbsDown, MessageSquare, Star } from "lucide-react"

interface ManualFeedbackModalProps {
  isOpen: boolean
  onClose: () => void
  onSubmit: (feedback: "up" | "down", text?: string, rating?: number) => void
  feedbackType: "up" | "down"
  messageContent: string
}

export default function ManualFeedbackModal({
  isOpen,
  onClose,
  onSubmit,
  feedbackType,
  messageContent,
}: ManualFeedbackModalProps) {
  const [feedbackText, setFeedbackText] = useState("")
  const [rating, setRating] = useState<number>(0)
  const [hoveredRating, setHoveredRating] = useState<number>(0)

  const handleSubmit = () => {
    onSubmit(feedbackType, feedbackText.trim() || undefined, rating || undefined)
    setFeedbackText("")
    setRating(0)
    setHoveredRating(0)
  }

  const handleSkip = () => {
    onSubmit(feedbackType)
    setFeedbackText("")
    setRating(0)
    setHoveredRating(0)
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl w-full max-w-md max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-2">
            {feedbackType === "up" ? (
              <ThumbsUp className="text-green-500" size={20} />
            ) : (
              <ThumbsDown className="text-red-500" size={20} />
            )}
            <h2 className="text-lg font-semibold text-gray-800 dark:text-gray-200">
              {feedbackType === "up" ? "Positive" : "Negative"} Feedback
            </h2>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors touch-manipulation"
          >
            <X size={20} className="text-gray-500 dark:text-gray-400" />
          </button>
        </div>

        {/* Content */}
        <div className="p-4 space-y-4">
          {/* AI Response Preview */}
          <div>
            <div className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">AI Response</div>
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3 max-h-32 overflow-y-auto">
              <p className="text-sm text-gray-900 dark:text-gray-100 break-words">{messageContent}</p>
            </div>
          </div>

          {/* Rating */}
          <div>
            <div className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Rate this response (optional)
            </div>
            <div className="flex gap-1">
              {[1, 2, 3, 4, 5].map((star) => (
                <button
                  key={star}
                  onClick={() => setRating(star)}
                  onMouseEnter={() => setHoveredRating(star)}
                  onMouseLeave={() => setHoveredRating(0)}
                  className="p-1 hover:scale-110 transition-transform touch-manipulation"
                >
                  <Star
                    size={24}
                    className={`${
                      star <= (hoveredRating || rating)
                        ? "text-yellow-400 fill-current"
                        : "text-gray-300 dark:text-gray-600"
                    } transition-colors`}
                  />
                </button>
              ))}
            </div>
            {rating > 0 && <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">{rating} out of 5 stars</div>}
          </div>

          {/* Text Feedback */}
          <div>
            <div className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Additional feedback (optional)
            </div>
            <div className="relative">
              <MessageSquare size={16} className="absolute left-3 top-3 text-gray-400 dark:text-gray-500" />
              <textarea
                value={feedbackText}
                onChange={(e) => setFeedbackText(e.target.value)}
                placeholder="Share your thoughts about this response..."
                className="w-full pl-10 pr-4 py-3 border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors text-sm md:text-base"
                rows={3}
                maxLength={500}
              />
            </div>
            <div className="text-xs text-gray-500 dark:text-gray-400 mt-1 text-right">{feedbackText.length}/500</div>
          </div>

          {/* Quick Emoji Reactions */}
          <div>
            <div className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Quick reactions (optional)</div>
            <div className="flex flex-wrap gap-2">
              {feedbackType === "up"
                ? ["👍", "🎉", "💯", "🔥", "✨", "👏"].map((emoji) => (
                    <button
                      key={emoji}
                      onClick={() => setFeedbackText((prev) => prev + emoji)}
                      className="p-2 text-lg hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors touch-manipulation"
                    >
                      {emoji}
                    </button>
                  ))
                : ["👎", "😕", "🤔", "❌", "⚠️", "💭"].map((emoji) => (
                    <button
                      key={emoji}
                      onClick={() => setFeedbackText((prev) => prev + emoji)}
                      className="p-2 text-lg hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors touch-manipulation"
                    >
                      {emoji}
                    </button>
                  ))}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex gap-3 p-4 border-t border-gray-200 dark:border-gray-700">
          <button
            onClick={handleSkip}
            className="flex-1 px-4 py-2 text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 rounded-lg transition-colors touch-manipulation"
          >
            Skip
          </button>
          <button
            onClick={handleSubmit}
            className={`flex-1 px-4 py-2 text-white rounded-lg transition-colors touch-manipulation ${
              feedbackType === "up" ? "bg-green-500 hover:bg-green-600" : "bg-red-500 hover:bg-red-600"
            }`}
          >
            Submit Feedback
          </button>
        </div>
      </div>
    </div>
  )
}
