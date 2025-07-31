"use client"

import { Menu } from "lucide-react"
import DarkModeToggle from "@/components/dark-mode-toggle"
import ModelSelector from "@/components/model-selector"
import type { RLModel } from "@/types"

interface TopBarProps {
  selectedModel: RLModel
  onModelChange: (model: RLModel) => void
  onMenuToggle: () => void
}

export default function TopBar({ selectedModel, onModelChange, onMenuToggle }: TopBarProps) {
  return (
    <header className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700 sticky top-0 z-50 transition-colors">
      <div className="max-w-7xl mx-auto px-4 py-3">
        <div className="flex items-center justify-between">
          {/* Left: Hamburger Menu */}
          <button
            onClick={onMenuToggle}
            className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors lg:hidden"
            aria-label="Toggle menu"
          >
            <Menu size={24} className="text-gray-600 dark:text-gray-300" />
          </button>

          {/* Center: App Title and Tagline */}
          <div className="flex-1 text-center lg:text-left lg:flex-none">
            <h1 className="text-xl font-semibold text-gray-800 dark:text-gray-200 transition-colors">ReinforceMe</h1>
            <p className="text-xs text-gray-500 dark:text-gray-400 hidden sm:block">Smarter with every prompt</p>
          </div>

          {/* Right: Model Selector & Dark Mode Toggle */}
          <div className="flex items-center gap-3">
            <ModelSelector selectedModel={selectedModel} onModelChange={onModelChange} />
            <DarkModeToggle />
          </div>
        </div>
      </div>
    </header>
  )
}
