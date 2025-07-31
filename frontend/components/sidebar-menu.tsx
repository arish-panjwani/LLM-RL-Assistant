"use client"

import { X, MessageCircle, Activity, FileText, Moon, Sun, Brain } from "lucide-react"
import { useTheme } from "@/components/theme-provider"
import type { RLModel } from "@/types"

interface SidebarMenuProps {
  isOpen: boolean
  onClose: () => void
  activeView: "chat" | "feedback" | "logs"
  onViewChange: (view: "chat" | "feedback" | "logs") => void
  selectedModel: RLModel
  onModelChange: (model: RLModel) => void
}

const models: { value: RLModel; label: string; description: string }[] = [
  { value: "PPO", label: "PPO", description: "Proximal Policy Optimization" },
  { value: "DDPG", label: "DDPG", description: "Deep Deterministic Policy Gradient" },
  { value: "A2C", label: "A2C", description: "Advantage Actor-Critic" },
  { value: "SAC", label: "SAC", description: "Soft Actor-Critic" },
]

export default function SidebarMenu({
  isOpen,
  onClose,
  activeView,
  onViewChange,
  selectedModel,
  onModelChange,
}: SidebarMenuProps) {
  const { theme, toggleTheme } = useTheme()

  const menuItems = [
    {
      id: "chat",
      label: "Chat",
      icon: MessageCircle,
      description: "Main conversation interface",
    },
    {
      id: "feedback",
      label: "Live Feedback Flow",
      icon: Activity,
      description: "Real-time AI feedback pipeline",
    },
    {
      id: "logs",
      label: "Prompt Logs",
      icon: FileText,
      description: "History of prompts and responses",
    },
  ]

  const handleViewChange = (view: "chat" | "feedback" | "logs") => {
    onViewChange(view)
    onClose()
  }

  return (
    <>
      {/* Sidebar - Fixed full viewport height */}
      <div
        className={`fixed top-0 left-0 w-80 max-w-[85vw] bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 transform transition-transform duration-300 ease-in-out z-40 ${
          isOpen ? "translate-x-0" : "-translate-x-full"
        } lg:relative lg:translate-x-0 lg:w-64 lg:max-w-none flex flex-col`}
        style={{ height: "100vh" }}
      >
        {/* Header - Fixed at top */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700 flex-shrink-0">
          <h2 className="text-lg font-semibold text-gray-800 dark:text-gray-200">Menu</h2>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors lg:hidden touch-manipulation"
          >
            <X size={20} className="text-gray-600 dark:text-gray-300" />
          </button>
        </div>

        {/* Navigation Items - Scrollable middle section */}
        <div className="flex-1 overflow-y-auto">
          <nav className="p-4 space-y-2">
            {menuItems.map((item) => {
              const Icon = item.icon
              return (
                <button
                  key={item.id}
                  onClick={() => handleViewChange(item.id as "chat" | "feedback" | "logs")}
                  className={`w-full flex items-center gap-3 p-3 md:p-4 rounded-lg transition-colors text-left touch-manipulation ${
                    activeView === item.id
                      ? "bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400"
                      : "text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
                  }`}
                >
                  <Icon size={20} className="flex-shrink-0" />
                  <div className="min-w-0 flex-1">
                    <div className="font-medium text-sm md:text-base break-words">{item.label}</div>
                    <div className="text-xs text-gray-500 dark:text-gray-400 break-words">{item.description}</div>
                  </div>
                </button>
              )
            })}
          </nav>
        </div>

        {/* Settings Section - Fixed at bottom with proper spacing */}
        <div className="border-t border-gray-200 dark:border-gray-700 p-3 md:p-4 space-y-3 md:space-y-4 flex-shrink-0 bg-white dark:bg-gray-800 mt-auto">
          {/* RL Model Selector */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              <Brain size={16} className="inline mr-2" />
              RL Model
            </label>
            <select
              value={selectedModel}
              onChange={(e) => onModelChange(e.target.value as RLModel)}
              className="w-full p-2 md:p-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 touch-manipulation"
            >
              {models.map((model) => (
                <option key={model.value} value={model.value}>
                  {model.label} - {model.description}
                </option>
              ))}
            </select>
          </div>

          {/* Dark Mode Toggle */}
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Dark Mode</span>
            <button
              onClick={toggleTheme}
              className="p-2 rounded-lg bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors touch-manipulation"
            >
              {theme === "light" ? (
                <Moon size={16} className="text-gray-600 dark:text-gray-300" />
              ) : (
                <Sun size={16} className="text-gray-600 dark:text-gray-300" />
              )}
            </button>
          </div>
        </div>
      </div>
    </>
  )
}
