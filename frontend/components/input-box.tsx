"use client"

import type React from "react"
import { useRef, useState } from "react"
import { Send, Camera, Mic, X } from "lucide-react"
import CameraModal from "@/components/camera-modal"

interface InputBoxProps {
  value: string
  onChange: (value: string) => void
  onSend: () => void
  onVoiceInput: () => void
  onImageUpload: (file: File) => void
  disabled?: boolean
  isListening?: boolean
}

export default function InputBox({
  value,
  onChange,
  onSend,
  onVoiceInput,
  onImageUpload,
  disabled = false,
  isListening = false,
}: InputBoxProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [showCameraModal, setShowCameraModal] = useState(false)
  const [speechError, setSpeechError] = useState<string | null>(null)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (value.trim() && !disabled) {
      onSend()
    }
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

  const handleVoiceInput = () => {
    setSpeechError(null)
    onVoiceInput()
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file && file.type.startsWith("image/")) {
      onImageUpload(file)
    }
    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }

  return (
    <div className="border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4 transition-colors">
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

      <form onSubmit={handleSubmit} className="flex items-end gap-2">
        {/* Hidden file input */}
        <input ref={fileInputRef} type="file" accept="image/*" onChange={handleFileChange} className="hidden" />

        {/* Input Actions */}
        <div className="flex gap-1">
          {/* Voice Input Button */}
          <button
            type="button"
            onClick={handleVoiceInput}
            disabled={disabled}
            className={`p-2 rounded-full transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
              isListening
                ? "bg-red-100 dark:bg-red-900/20 text-red-600 dark:text-red-400 animate-pulse"
                : "text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
            }`}
            title={isListening ? "Stop Recording" : "Voice Input"}
          >
            <Mic size={20} />
          </button>

          {/* Image Upload Button */}
          <button
            type="button"
            onClick={handleCameraClick}
            disabled={disabled}
            className="p-2 text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-full transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            title="Camera & Upload"
          >
            <Camera size={20} />
          </button>
        </div>

        {/* Text Input */}
        <div className="flex-1 relative">
          <textarea
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={isListening ? "Listening..." : "Type your message..."}
            disabled={disabled}
            className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            rows={1}
            style={{ minHeight: "40px", maxHeight: "120px" }}
          />
        </div>

        {/* Send Button */}
        <button
          type="submit"
          disabled={!value.trim() || disabled}
          className="p-2 bg-blue-500 text-white rounded-full hover:bg-blue-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-blue-500"
          title="Send Message"
        >
          <Send size={20} />
        </button>
      </form>
      {/* Camera Modal */}
      <CameraModal
        isOpen={showCameraModal}
        onClose={() => setShowCameraModal(false)}
        onCapture={(file) => onImageUpload(file)}
        onUpload={handleImageUpload}
      />
    </div>
  )
}
