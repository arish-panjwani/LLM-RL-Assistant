"use client"

import { useState, useRef, useEffect } from "react"
import { ChevronDown, Brain } from "lucide-react"
import type { RLModel } from "@/types"

interface ModelSelectorProps {
  selectedModel: RLModel
  onModelChange: (model: RLModel) => void
}

const models: { value: RLModel; label: string; description: string }[] = [
  { value: "PPO", label: "PPO", description: "Proximal Policy Optimization" },
  { value: "DDPG", label: "DDPG", description: "Deep Deterministic Policy Gradient" },
  { value: "A2C", label: "A2C", description: "Advantage Actor-Critic" },
  { value: "SAC", label: "SAC", description: "Soft Actor-Critic" },
]

export default function ModelSelector({ selectedModel, onModelChange }: ModelSelectorProps) {
  const [isOpen, setIsOpen] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false)
      }
    }

    document.addEventListener("mousedown", handleClickOutside)
    return () => document.removeEventListener("mousedown", handleClickOutside)
  }, [])

  const handleModelSelect = (model: RLModel) => {
    onModelChange(model)
    setIsOpen(false)
    // TODO: Send model selection to Flask backend
    console.log("Selected RL model:", model)
  }

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-3 py-2 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 rounded-lg transition-colors text-sm font-medium text-gray-700 dark:text-gray-300"
      >
        <Brain size={16} />
        <span className="hidden sm:inline">{selectedModel}</span>
        <ChevronDown size={16} className={`transition-transform ${isOpen ? "rotate-180" : ""}`} />
      </button>

      {isOpen && (
        <div className="absolute right-0 mt-2 w-64 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg z-50">
          <div className="py-1">
            {models.map((model) => (
              <button
                key={model.value}
                onClick={() => handleModelSelect(model.value)}
                className={`w-full text-left px-4 py-3 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors ${
                  selectedModel === model.value
                    ? "bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400"
                    : "text-gray-700 dark:text-gray-300"
                }`}
              >
                <div className="font-medium">{model.label}</div>
                <div className="text-xs text-gray-500 dark:text-gray-400">{model.description}</div>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
