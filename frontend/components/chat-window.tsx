"use client"

import { useState, useRef, useEffect } from "react"
import MessageBubble from "@/components/message-bubble"
import ChatInput from "@/components/chat-input"
import { RPCHelper } from "@/utils/rpc-helper"
import type { RLModel, Message } from "@/types"

interface ChatWindowProps {
  selectedModel: RLModel
}

export default function ChatWindow({ selectedModel }: ChatWindowProps) {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      content: "Hello! I'm your AI assistant. How can I help you today?",
      sender: "assistant",
      timestamp: new Date(),
      feedback: null,
    },
  ])
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const sendMessage = async (content: string, type: "text" | "image" = "text", imageFile?: File) => {
    if (!content.trim() && !imageFile) return

    // Create image URL for display if image is provided
    let imageUrl: string | undefined
    if (imageFile) {
      imageUrl = URL.createObjectURL(imageFile)
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      content,
      sender: "user",
      timestamp: new Date(),
      feedback: null,
      originalPrompt: content,
      imageUrl: imageUrl,
      imageFile: imageFile,
    }

    setMessages((prev) => [...prev, userMessage])
    setIsLoading(true)

    try {
      // Process prompt and get AI response
      const { response, modifiedPrompt } = await RPCHelper.sendMessage(
        imageFile ? `[Image: ${imageFile.name}] ${content}` : content,
        selectedModel,
      )

      const aiResponse: Message = {
        id: (Date.now() + 1).toString(),
        content: response,
        sender: "assistant",
        timestamp: new Date(),
        feedback: null,
        originalPrompt: content,
        modifiedPrompt: modifiedPrompt,
      }

      setMessages((prev) => [...prev, aiResponse])
    } catch (error) {
      console.error("Error sending message:", error)
      // Add error message
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: "Sorry, I encountered an error. Please try again.",
        sender: "assistant",
        timestamp: new Date(),
        feedback: null,
      }
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleFeedback = async (messageId: string, feedback: "up" | "down") => {
    setMessages((prev) => prev.map((msg) => (msg.id === messageId ? { ...msg, feedback } : msg)))

    try {
      await RPCHelper.sendFeedback(messageId, feedback, selectedModel)
    } catch (error) {
      console.error("Error sending feedback:", error)
    }
  }

  const handleTTS = (text: string) => {
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel()

      const utterance = new SpeechSynthesisUtterance(text)

      // Siri-like voice settings
      utterance.rate = 0.85 // Slightly slower, more natural
      utterance.pitch = 1.1 // Slightly higher pitch
      utterance.volume = 0.9 // Slightly softer volume

      // Wait for voices to load and select the best one
      const setVoice = () => {
        const voices = window.speechSynthesis.getVoices()

        // Prefer high-quality voices (usually contain "Enhanced", "Premium", or specific names)
        const preferredVoices = [
          // iOS/Safari voices
          "Samantha",
          "Alex",
          "Victoria",
          "Karen",
          "Moira",
          // Chrome/Edge enhanced voices
          "Google US English",
          "Microsoft Zira Desktop",
          "Microsoft David Desktop",
          // Look for enhanced/premium voices
          ...voices
            .filter(
              (voice) =>
                voice.name.includes("Enhanced") || voice.name.includes("Premium") || voice.name.includes("Neural"),
            )
            .map((v) => v.name),
        ]

        // Find the best available voice
        let selectedVoice = null
        for (const preferredName of preferredVoices) {
          selectedVoice = voices.find((voice) => voice.name === preferredName && voice.lang.startsWith("en"))
          if (selectedVoice) break
        }

        // Fallback to any high-quality English voice
        if (!selectedVoice) {
          selectedVoice = voices.find(
            (voice) =>
              voice.lang.startsWith("en") &&
              (voice.localService || voice.name.includes("Google") || voice.name.includes("Microsoft")),
          )
        }

        // Final fallback to any English voice
        if (!selectedVoice) {
          selectedVoice = voices.find((voice) => voice.lang.startsWith("en"))
        }

        if (selectedVoice) {
          utterance.voice = selectedVoice
        }

        window.speechSynthesis.speak(utterance)
      }

      // Check if voices are already loaded
      if (window.speechSynthesis.getVoices().length > 0) {
        setVoice()
      } else {
        // Wait for voices to load
        window.speechSynthesis.onvoiceschanged = setVoice
      }
    } else {
      console.warn("Text-to-Speech not supported in this browser")
    }
  }

  return (
    <div className="flex flex-col h-full ">
      {/* Chat Messages - Flexible height */}
      <div className="flex-1 overflow-y-auto p-3 md:p-4 space-y-3 md:space-y-4">
        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} onFeedback={handleFeedback} onTTS={handleTTS} />
        ))}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-gray-200 dark:bg-gray-700 rounded-lg px-3 md:px-4 py-2 max-w-xs">
              <div className="flex space-x-1">
                <div className="w-2 h-2 bg-gray-400 dark:bg-gray-500 rounded-full animate-bounce"></div>
                <div
                  className="w-2 h-2 bg-gray-400 dark:bg-gray-500 rounded-full animate-bounce"
                  style={{ animationDelay: "0.1s" }}
                ></div>
                <div
                  className="w-2 h-2 bg-gray-400 dark:bg-gray-500 rounded-full animate-bounce"
                  style={{ animationDelay: "0.2s" }}
                ></div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Chat Input - Fixed at bottom */}
      <div className="flex-shrink-0">
        <ChatInput onSendMessage={sendMessage} disabled={isLoading} />
      </div>
    </div>
  )
}
