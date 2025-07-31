"use client"

import { useState, useRef, useEffect } from "react"
import MessageBubble from "@/components/message-bubble"
import InputBox from "@/components/input-box"
import type { RLModel } from "@/app/page"

type SpeechRecognitionType = InstanceType<typeof window.SpeechRecognition | typeof window.webkitSpeechRecognition>


export interface Message {
  id: string
  content: string
  sender: "user" | "assistant"
  timestamp: Date
  feedback?: "up" | "down" | null
  hasFeedbackRequest?: boolean
}

interface ChatTabProps {
  selectedModel: RLModel
}

export default function ChatTab({ selectedModel }: ChatTabProps) {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      content: "Hello! I'm your AI assistant. How can I help you today?",
      sender: "assistant",
      timestamp: new Date(),
      feedback: null,
      hasFeedbackRequest: true,
    },
  ])
  const [inputText, setInputText] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [isListening, setIsListening] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  type SpeechRecognitionType = InstanceType<typeof window.SpeechRecognition | typeof window.webkitSpeechRecognition>
  const recognitionRef = useRef<SpeechRecognitionType | null>(null)


  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Initialize Speech Recognition
  useEffect(() => {
    if (typeof window !== "undefined" && ("webkitSpeechRecognition" in window || "SpeechRecognition" in window)) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
      recognitionRef.current = new SpeechRecognition()

      if (recognitionRef.current) {
        recognitionRef.current.continuous = false
        recognitionRef.current.interimResults = false
        recognitionRef.current.lang = "en-US"

        recognitionRef.current.onresult = (event) => {
          const transcript = event.results[0][0].transcript
          setInputText(transcript)
          setIsListening(false)
        }

        recognitionRef.current.onerror = (event) => {
          console.error("Speech recognition error:", event.error)
          setIsListening(false)
        }

        recognitionRef.current.onend = () => {
          setIsListening(false)
        }
      }
    }
  }, [])

  // TODO: Integrate with Flask API
  const sendMessage = async (content: string, type: "text" | "image" = "text") => {
    if (!content.trim()) return

    const userMessage: Message = {
      id: Date.now().toString(),
      content,
      sender: "user",
      timestamp: new Date(),
      feedback: null,
    }

    setMessages((prev) => [...prev, userMessage])
    setInputText("")
    setIsLoading(true)

    // TODO: Replace with actual Flask API call using selectedModel
    setTimeout(() => {
      const aiResponse: Message = {
        id: (Date.now() + 1).toString(),
        content: `I received your message: "${content}". This response was generated using the ${selectedModel} model. This is a placeholder response. In the next phase, I'll be connected to a Flask backend with WebSocket for real-time communication.`,
        sender: "assistant",
        timestamp: new Date(),
        feedback: null,
        hasFeedbackRequest: true,
      }
      setMessages((prev) => [...prev, aiResponse])
      setIsLoading(false)
    }, 1000)
  }

  // TODO: Integrate with WebSocket for real-time responses
  const receiveMessage = (content: string) => {
    // WebSocket message handler placeholder
    console.log("Received message via WebSocket:", content)
  }

  // TODO: Send feedback to Flask API
  const sendFeedback = async (messageId: string, feedback: "up" | "down") => {
    setMessages((prev) => prev.map((msg) => (msg.id === messageId ? { ...msg, feedback } : msg)))
    // TODO: Send feedback to backend with selectedModel context
    console.log("Sending feedback:", { messageId, feedback, model: selectedModel })
  }

  // TODO: Get AI feedback from Flask API
  const getAIFeedback = async (messageId: string) => {
    // TODO: Request AI feedback analysis from backend
    console.log("Requesting AI feedback for message:", messageId, "using model:", selectedModel)
    // Placeholder: This would trigger backend analysis of the AI response quality
  }

  const handleFeedback = (messageId: string, feedback: "up" | "down") => {
    sendFeedback(messageId, feedback)
  }

  const handleTTS = (text: string) => {
    if ("speechSynthesis" in window) {
      // Cancel any ongoing speech
      window.speechSynthesis.cancel()

      const utterance = new SpeechSynthesisUtterance(text)
      utterance.rate = 0.9
      utterance.pitch = 1
      utterance.volume = 1

      // Get available voices and prefer English voices
      const voices = window.speechSynthesis.getVoices()
      const englishVoice = voices.find((voice) => voice.lang.startsWith("en"))
      if (englishVoice) {
        utterance.voice = englishVoice
      }

      window.speechSynthesis.speak(utterance)
    } else {
      console.warn("Text-to-Speech not supported in this browser")
    }
  }

  const handleVoiceInput = () => {
    if (!recognitionRef.current) {
      alert("Speech recognition is not supported in this browser")
      return
    }

    if (isListening) {
      recognitionRef.current.stop()
      setIsListening(false)
    } else {
      setIsListening(true)
      recognitionRef.current.start()
    }
  }

  const handleImageUpload = (file: File) => {
    // TODO: Process image upload with selectedModel
    console.log("Image uploaded:", file.name, "Model:", selectedModel)
    sendMessage(`[Image uploaded: ${file.name}]`, "image")
  }

  return (
    <div className="flex flex-col h-[calc(100vh-144px)]">
      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <MessageBubble
            key={message.id}
            message={message}
            onFeedback={handleFeedback}
            onTTS={handleTTS}
            onGetAIFeedback={getAIFeedback}
          />
        ))}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-gray-200 dark:bg-gray-700 rounded-lg px-4 py-2 max-w-xs">
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

      {/* Input Area */}
      <InputBox
        value={inputText}
        onChange={setInputText}
        onSend={() => sendMessage(inputText)}
        onVoiceInput={handleVoiceInput}
        onImageUpload={handleImageUpload}
        disabled={isLoading}
        isListening={isListening}
      />
    </div>
  )
}
