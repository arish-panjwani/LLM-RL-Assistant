"use client"

import { useEffect, useRef, useState } from "react"
import { Mic } from "lucide-react"

interface SpeechInputProps {
  isListening: boolean
  onStart: () => void
  onStop: () => void
  onResult: (transcript: string) => void
  onError: (error: string) => void
  disabled?: boolean
}

export default function SpeechInput({
  isListening,
  onStart,
  onStop,
  onResult,
  onError,
  disabled = false,
}: SpeechInputProps) {
  const recognitionRef = useRef<any>(null)
  const [isSupported, setIsSupported] = useState(false)
  const isStoppingRef = useRef(false)

  useEffect(() => {
    if (typeof window !== "undefined" && ("webkitSpeechRecognition" in window || "SpeechRecognition" in window)) {
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
      recognitionRef.current = new SpeechRecognition()
      setIsSupported(true)

      if (recognitionRef.current) {
        recognitionRef.current.continuous = false
        recognitionRef.current.interimResults = false
        recognitionRef.current.lang = "en-US"

        recognitionRef.current.onresult = (event: any) => {
          const transcript = event.results[0][0].transcript
          onResult(transcript)
          isStoppingRef.current = false
        }

        recognitionRef.current.onerror = (event: any) => {
          // Don't treat "aborted" as an error since it's expected when we stop manually
          if (event.error === "aborted" && isStoppingRef.current) {
            isStoppingRef.current = false
            return
          }

          // Handle other errors appropriately
          if (event.error === "no-speech") {
            console.log("No speech detected, please try again")
          } else if (event.error === "network") {
            onError("Network error occurred during speech recognition")
          } else if (event.error === "not-allowed") {
            onError("Microphone access denied. Please allow microphone access and try again.")
          } else if (event.error !== "aborted") {
            onError(`Speech recognition error: ${event.error}`)
          }

          isStoppingRef.current = false
        }

        recognitionRef.current.onend = () => {
          if (!isStoppingRef.current) {
            onStop()
          }
          isStoppingRef.current = false
        }

        recognitionRef.current.onstart = () => {
          isStoppingRef.current = false
        }
      }
    }
  }, [onResult, onError, onStop])

  const handleClick = () => {
    if (!recognitionRef.current || !isSupported) {
      onError("Speech recognition is not supported in this browser")
      return
    }

    if (isListening) {
      // Set flag to indicate we're stopping manually
      isStoppingRef.current = true
      try {
        recognitionRef.current.stop()
      } catch (error) {
        console.warn("Error stopping speech recognition:", error)
      }
      onStop()
    } else {
      try {
        isStoppingRef.current = false
        onStart()
        recognitionRef.current.start()
      } catch (error) {
        console.error("Error starting speech recognition:", error)
        onError("Failed to start speech recognition")
      }
    }
  }

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (recognitionRef.current && isListening) {
        isStoppingRef.current = true
        try {
          recognitionRef.current.stop()
        } catch (error) {
          console.warn("Error stopping speech recognition on cleanup:", error)
        }
      }
    }
  }, [isListening])

  if (!isSupported) {
    return null // Don't render if not supported
  }

  return (
    <button
      type="button"
      onClick={handleClick}
      disabled={disabled}
      className={`p-2 md:p-3 rounded-full transition-colors disabled:opacity-50 disabled:cursor-not-allowed touch-manipulation ${
        isListening
          ? "bg-red-100 dark:bg-red-900/20 text-red-600 dark:text-red-400 animate-pulse"
          : "text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
      }`}
      title={isListening ? "Stop Recording" : "Voice Input"}
    >
      <Mic size={18} />
    </button>
  )
}
