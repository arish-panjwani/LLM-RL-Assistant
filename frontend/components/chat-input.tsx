"use client"

import type React from "react"

import { useState, useRef } from "react"
import { Send, Camera, X } from "lucide-react"
import SpeechInput from "@/components/speech-input"
import CameraModal from "@/components/camera-modal"

interface ChatInputProps {
  onSendMessage: (message: string, type?: "text" | "image", imageFile?: File) => void
  disabled?: boolean
}

export default function ChatInput({ onSendMessage, disabled = false }: ChatInputProps) {
  const [inputText, setInputText] = useState("")
  const [isListening, setIsListening] = useState(false)
  const [selectedImage, setSelectedImage] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [speechError, setSpeechError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [showCameraModal, setShowCameraModal] = useState(false)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (disabled) return

    if (selectedImage) {
      // Send image with optional text
      onSendMessage(inputText.trim() || "Image", "image", selectedImage)
      clearImage()
    } else if (inputText.trim()) {
      // Send text only
      onSendMessage(inputText)
    }

    setInputText("")
    setSpeechError(null)
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  const handleCameraClick = () => {
    setShowCameraModal(true)
  }

  const handleImageUpload = () => {
    fileInputRef.current?.click()
  }

  const handleCameraCapture = (file: File) => {
    setSelectedImage(file)
    createImagePreview(file)
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file && file.type.startsWith("image/")) {
      setSelectedImage(file)
      createImagePreview(file)
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }

  const createImagePreview = (file: File) => {
    const reader = new FileReader()
    reader.onload = (e) => {
      setImagePreview(e.target?.result as string)
    }
    reader.readAsDataURL(file)
  }

  const clearImage = () => {
    setSelectedImage(null)
    setImagePreview(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }

  const handleSpeechStart = () => {
    setIsListening(true)
    setSpeechError(null)
  }

  const handleSpeechStop = () => {
    setIsListening(false)
  }

  const handleSpeechResult = (transcript: string) => {
    setInputText(transcript)
    setIsListening(false)
    setSpeechError(null)
  }

  const handleSpeechError = (error: string) => {
    console.warn("Speech recognition error:", error)
    setSpeechError(error)
    setIsListening(false)

    // Auto-clear error after 3 seconds
    setTimeout(() => {
      setSpeechError(null)
    }, 3000)
  }

  return (
    <div className="border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 transition-colors">
      <div className="p-3 md:p-4">
        {/* Speech Error Display */}
        {speechError && (
          <div className="mb-2 p-2 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
            <div className="flex items-center justify-between">
              <p className="text-sm text-red-600 dark:text-red-400">{speechError}</p>
              <button
                onClick={() => setSpeechError(null)}
                className="text-red-400 hover:text-red-600 dark:hover:text-red-300"
              >
                <X size={14} />
              </button>
            </div>
          </div>
        )}

        {/* Image Preview */}
        {imagePreview && (
          <div className="mb-3 relative inline-block">
            <div className="relative">
              <img
                src={imagePreview || "/placeholder.svg"}
                alt="Selected image"
                className="max-w-32 max-h-32 rounded-lg border border-gray-300 dark:border-gray-600 object-cover"
              />
              <button
                onClick={clearImage}
                className="absolute -top-2 -right-2 p-1 bg-red-500 text-white rounded-full hover:bg-red-600 transition-colors touch-manipulation"
                title="Remove image"
              >
                <X size={14} />
              </button>
            </div>
            <div className="text-xs text-gray-500 dark:text-gray-400 mt-1 max-w-32 truncate">{selectedImage?.name}</div>
          </div>
        )}

        <form onSubmit={handleSubmit} className="flex items-center gap-2">
          {/* Hidden file input */}
          <input ref={fileInputRef} type="file" accept="image/*" onChange={handleFileChange} className="hidden" />

          {/* Input Actions */}
          <div className="flex gap-1 flex-shrink-0">
            {/* Speech Input */}
            <SpeechInput
              isListening={isListening}
              onStart={handleSpeechStart}
              onStop={handleSpeechStop}
              onResult={handleSpeechResult}
              onError={handleSpeechError}
              disabled={disabled}
            />

            {/* Camera Button */}
            <button
              type="button"
              onClick={handleCameraClick}
              disabled={disabled}
              className="p-2 md:p-3 text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-full transition-colors disabled:opacity-50 disabled:cursor-not-allowed touch-manipulation"
              title="Camera & Upload"
            >
              <Camera size={18} />
            </button>
          </div>

          {/* Text Input */}
          <div className="flex-1 relative min-w-0">
            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={
                isListening
                  ? "Listening..."
                  : selectedImage
                    ? "Add a message with your image (optional)..."
                    : "Type your message..."
              }
              disabled={disabled}
              className="w-full px-3 md:px-4 py-2 md:py-3 border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-sm md:text-base"
              rows={1}
              style={{ minHeight: "44px", maxHeight: "120px" }}
            />
          </div>

          {/* Send Button */}
          <button
            type="submit"
            disabled={(!inputText.trim() && !selectedImage) || disabled}
            className="flex-shrink-0 p-2 md:p-3 bg-blue-500 text-white rounded-full hover:bg-blue-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-blue-500 touch-manipulation"
            title="Send Message"
          >
            <Send size={18} />
          </button>
        </form>
      </div>

      {/* Camera Modal */}
      <CameraModal
        isOpen={showCameraModal}
        onClose={() => setShowCameraModal(false)}
        onCapture={handleCameraCapture}
        onUpload={handleImageUpload}
      />
    </div>
  )
}
