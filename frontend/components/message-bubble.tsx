"use client"

import { ThumbsUp, ThumbsDown, Volume2, MessageSquare } from "lucide-react"
import { useRouter } from "next/navigation"
import { useState } from "react"
import type { Message } from "@/types"
import ManualFeedbackModal from "@/components/manual-feedback-modal"

interface MessageBubbleProps {
  message: Message
  onFeedback: (messageId: string, feedback: "up" | "down") => void
  onTTS: (text: string) => void
}

export default function MessageBubble({ message, onFeedback, onTTS }: MessageBubbleProps) {
  const router = useRouter()
  const isUser = message.sender === "user"

  const [showFeedbackModal, setShowFeedbackModal] = useState(false)
  const [pendingFeedback, setPendingFeedback] = useState<"up" | "down" | null>(null)

  const formatTime = (date: Date) => {
    const hours = date.getUTCHours().toString().padStart(2, "0")
    const minutes = date.getUTCMinutes().toString().padStart(2, "0")
    return `${hours}:${minutes}`
  }

  const handleGetFeedback = () => {
    console.log("Getting feedback for message:", message.id)
  }

  const handleFeedbackClick = (feedback: "up" | "down") => {
    setPendingFeedback(feedback)
    setShowFeedbackModal(true)
  }

  const handleFeedbackSubmit = (feedback: "up" | "down", text?: string, rating?: number) => {
    onFeedback(message.id, feedback)

    if ((window as any).setLiveFeedbackContext) {
      ;(window as any).setLiveFeedbackContext({
        originalPrompt: message.originalPrompt || "No original prompt available",
        aiResponse: message.content,
        manualFeedback: feedback,
        manualFeedbackText: text,
      })
    }

    if ((window as any).navigateToFeedbackFlow) {
      ;(window as any).navigateToFeedbackFlow()
    }

    setShowFeedbackModal(false)
    setPendingFeedback(null)
  }

  const handleModalClose = () => {
    setShowFeedbackModal(false)
    setPendingFeedback(null)
  }

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`max-w-xs sm:max-w-md lg:max-w-lg ${isUser ? "order-2" : "order-1"}`}>
        {/* User Type Badge */}
        <div className={`text-xs text-gray-500 dark:text-gray-400 mb-1 ${isUser ? "text-right" : "text-left"}`}>
          {isUser ? "User" : "Assistant"}
        </div>

        {/* Message Content */}
        <div
          className={`rounded-lg px-3 md:px-4 py-2 md:py-3 ${
            isUser
              ? "bg-blue-500 text-white rounded-br-sm"
              : "bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 rounded-bl-sm"
          }`}
        >
          {/* Image Display */}
          {message.imageUrl && (
            <div className="mb-2">
              <img
                src={message.imageUrl || "/placeholder.svg"}
                alt="Sent image"
                className="max-w-full h-auto rounded-lg border border-white/20"
                style={{ maxHeight: "200px" }}
              />
            </div>
          )}

          {/* Text Content */}
          <p className="text-sm sm:text-base break-words">{message.content}</p>
        </div>

        {/* Timestamp and Actions */}
        <div
          className={`flex items-center gap-2 mt-1 text-xs text-gray-500 dark:text-gray-400 ${
            isUser ? "justify-end" : "justify-start"
          }`}
        >

          <span>{formatTime(new Date(message.timestamp))}</span>

          {/* AI Message Actions */}
          {!isUser && (
            <div className="flex items-center gap-1">
              {/* TTS Button */}
              <button
                onClick={() => onTTS(message.content)}
                className="p-1 hover:bg-gray-100 dark:hover:bg-gray-600 rounded transition-colors touch-manipulation"
                title="Text to Speech"
              >
                <Volume2 size={14} />
              </button>

              {/* Get Feedback Button */}
              <button
                onClick={handleGetFeedback}
                className="p-1 hover:bg-gray-100 dark:hover:bg-gray-600 rounded transition-colors text-purple-600 dark:text-purple-400 touch-manipulation"
                title="Get AI Feedback"
              >
                <MessageSquare size={14} />
              </button>

              {/* Feedback Buttons */}
              <button
                onClick={() => handleFeedbackClick("up")}
                className={`p-1 hover:bg-gray-100 dark:hover:bg-gray-600 rounded transition-colors touch-manipulation ${
                  message.feedback === "up" ? "text-green-600 dark:text-green-400" : ""
                }`}
                title="Thumbs Up"
              >
                <ThumbsUp size={14} />
              </button>
              <button
                onClick={() => handleFeedbackClick("down")}
                className={`p-1 hover:bg-gray-100 dark:hover:bg-gray-600 rounded transition-colors touch-manipulation ${
                  message.feedback === "down" ? "text-red-600 dark:text-red-400" : ""
                }`}
                title="Thumbs Down"
              >
                <ThumbsDown size={14} />
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Manual Feedback Modal */}
      {showFeedbackModal && pendingFeedback && (
        <ManualFeedbackModal
          isOpen={showFeedbackModal}
          onClose={handleModalClose}
          onSubmit={handleFeedbackSubmit}
          feedbackType={pendingFeedback}
          messageContent={message.content}
          isAutoTriggered={false}
        />
      )}
    </div>
  )
}
