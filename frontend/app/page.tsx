"use client"

import { useState, useEffect } from "react"
import { ThemeProvider } from "@/components/theme-provider"
import TopBar from "@/components/top-bar"
import SidebarMenu from "@/components/sidebar-menu"
import ChatWindow from "@/components/chat-window"
import LiveFeedbackFlow from "@/components/live-feedback-flow"
import PromptLogsView from "@/components/prompt-logs-view"
import type { RLModel } from "@/types"

export type { RLModel }

export default function ReinforceMe() {
  const [selectedModel, setSelectedModel] = useState<RLModel>("PPO")
  const [activeView, setActiveView] = useState<"chat" | "feedback" | "logs">("chat")
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)

  useEffect(() => {
    // Expose navigation function for feedback flow integration
    ;(window as any).navigateToFeedbackFlow = () => {
      setActiveView("feedback")
      setIsSidebarOpen(false)
    }
  }, [])

  const renderActiveView = () => {
    switch (activeView) {
      case "chat":
        return <ChatWindow selectedModel={selectedModel} />
      case "feedback":
        return <LiveFeedbackFlow />
      case "logs":
        return <PromptLogsView />
      default:
        return <ChatWindow selectedModel={selectedModel} />
    }
  }

  return (
    <ThemeProvider>
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex flex-col transition-colors">
        {/* Top Bar */}
        <TopBar
          selectedModel={selectedModel}
          onModelChange={setSelectedModel}
          onMenuToggle={() => setIsSidebarOpen(!isSidebarOpen)}
        />

        <div className="flex flex-1 relative overflow-hidden">
          {/* Sidebar Menu */}
          <SidebarMenu
            isOpen={isSidebarOpen}
            onClose={() => setIsSidebarOpen(false)}
            activeView={activeView}
            onViewChange={setActiveView}
            selectedModel={selectedModel}
            onModelChange={setSelectedModel}
          />

          {/* Main Content */}
          <main className="flex-1 relative overflow-hidden">
            {/* Overlay for mobile when sidebar is open */}
            {isSidebarOpen && (
              <div
                className="fixed inset-0 bg-black bg-opacity-50 z-30 lg:hidden"
                onClick={() => setIsSidebarOpen(false)}
              />
            )}
            <div className="h-full overflow-y-auto">{renderActiveView()}</div>
          </main>
        </div>
      </div>
    </ThemeProvider>
  )
}
