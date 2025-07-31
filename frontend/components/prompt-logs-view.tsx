"use client"

import { useState, useEffect } from "react"
import { Calendar, Brain, TrendingUp, TrendingDown, Filter } from "lucide-react"
import { getMockPromptLogs } from "@/mock/mock-data"
import type { PromptLog, RLModel } from "@/types"

// Add line-clamp utility styles
const lineClampStyles = `
  .line-clamp-3 {
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }
`

export default function PromptLogsView() {
  const [logs, setLogs] = useState<PromptLog[]>([])
  const [sortBy, setSortBy] = useState<"timestamp" | "reward">("timestamp")
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc")
  const [filterByModel, setFilterByModel] = useState<RLModel | "all">("all")

  useEffect(() => {
    // Load mock data
    setLogs(getMockPromptLogs())
  }, [])

  // Filter logs by model
  const filteredLogs = logs.filter((log) => {
    if (filterByModel === "all") return true
    return log.model === filterByModel
  })

  // Sort filtered logs
  const sortedLogs = [...filteredLogs].sort((a, b) => {
    let comparison = 0

    if (sortBy === "timestamp") {
      comparison = a.timestamp.getTime() - b.timestamp.getTime()
    } else if (sortBy === "reward") {
      comparison = a.finalReward - b.finalReward
    }

    return sortOrder === "asc" ? comparison : -comparison
  })

  const formatTimestamp = (date: Date) => {
    return date.toLocaleString()
  }

  const getRewardColor = (reward: number) => {
    if (reward >= 0.8) return "text-green-600 dark:text-green-400"
    if (reward >= 0.6) return "text-yellow-600 dark:text-yellow-400"
    return "text-red-600 dark:text-red-400"
  }

  const getModelBadgeColor = (model: string) => {
    const colors = {
      PPO: "bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400",
      DDPG: "bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400",
      A2C: "bg-purple-100 text-purple-800 dark:bg-purple-900/20 dark:text-purple-400",
      SAC: "bg-orange-100 text-orange-800 dark:bg-orange-900/20 dark:text-orange-400",
    }
    return colors[model as keyof typeof colors] || colors.PPO
  }

  const modelOptions = [
    { value: "all", label: "All Models" },
    { value: "PPO", label: "PPO" },
    { value: "DDPG", label: "DDPG" },
    { value: "A2C", label: "A2C" },
    { value: "SAC", label: "SAC" },
  ]

  return (
    <div className="h-full flex flex-col">
      <div className="flex-1 overflow-y-auto p-6 max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-2xl font-bold text-gray-800 dark:text-gray-200 mb-2">Prompt Logs</h1>
          <p className="text-gray-600 dark:text-gray-400">History of prompts, responses, and feedback evaluations</p>
        </div>

        {/* Controls */}
        <div className="mb-6 flex flex-wrap gap-4 items-center">
          {/* Sort By */}
          <div className="flex items-center gap-2">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">Sort by:</label>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as "timestamp" | "reward")}
              className="px-3 py-1 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="timestamp">Timestamp</option>
              <option value="reward">Reward Score</option>
            </select>
          </div>

          {/* Sort Order */}
          <button
            onClick={() => setSortOrder(sortOrder === "asc" ? "desc" : "asc")}
            className="flex items-center gap-1 px-3 py-1 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200 transition-colors"
          >
            {sortOrder === "asc" ? <TrendingUp size={16} /> : <TrendingDown size={16} />}
            {sortOrder === "asc" ? "Ascending" : "Descending"}
          </button>

          {/* Filter By Model */}
          <div className="flex items-center gap-2">
            <Filter size={16} className="text-gray-500 dark:text-gray-400" />
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">Filter by:</label>
            <select
              value={filterByModel}
              onChange={(e) => setFilterByModel(e.target.value as RLModel | "all")}
              className="px-3 py-1 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {modelOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          {/* Results Count */}
          <div className="text-sm text-gray-500 dark:text-gray-400 ml-auto">
            Showing {sortedLogs.length} of {logs.length} logs
          </div>
        </div>

        {/* Logs Table - Mobile First Responsive */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
          {/* Mobile View - Card Layout */}
          <div className="block md:hidden">
            {sortedLogs.map((log) => (
              <div key={log.id} className="border-b border-gray-200 dark:border-gray-700 last:border-b-0 p-4">
                <div className="space-y-3">
                  {/* Header with timestamp and model */}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
                      <Calendar size={14} />
                      <span className="text-xs">{formatTimestamp(log.timestamp)}</span>
                    </div>
                    <span
                      className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${getModelBadgeColor(log.model)}`}
                    >
                      <Brain size={12} />
                      {log.model}
                    </span>
                  </div>

                  {/* Prompts */}
                  <div className="space-y-2">
                    <div>
                      <div className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Original Prompt</div>
                      <div className="text-sm text-gray-900 dark:text-gray-100 break-words">{log.originalPrompt}</div>
                    </div>
                    <div>
                      <div className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Modified Prompt</div>
                      <div className="text-sm text-gray-600 dark:text-gray-400 break-words">{log.modifiedPrompt}</div>
                    </div>
                  </div>

                  {/* AI Response */}
                  <div>
                    <div className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">AI Response</div>
                    <div className="text-sm text-gray-900 dark:text-gray-100 break-words line-clamp-3">
                      {log.aiResponse}
                    </div>
                  </div>

                  {/* Feedback Summary and Reward */}
                  <div className="flex items-center justify-between pt-2 border-t border-gray-100 dark:border-gray-700">
                    <div className="text-xs space-y-1">
                      <div className="flex gap-4">
                        <span>Clarity: {log.feedbackSummary.promptClarity.toFixed(1)}/10</span>
                        <span>Consistency: {log.feedbackSummary.responseConsistency.toFixed(2)}</span>
                      </div>
                      <div className="flex gap-4">
                        <span>Sentiment: {log.feedbackSummary.sentimentScore.toFixed(2)}</span>
                        <span
                          className={
                            log.feedbackSummary.hallucinationFlag
                              ? "text-red-600 dark:text-red-400"
                              : "text-green-600 dark:text-green-400"
                          }
                        >
                          Hallucination: {log.feedbackSummary.hallucinationFlag ? "Yes" : "No"}
                        </span>
                      </div>
                    </div>
                    <div className={`text-lg font-semibold ${getRewardColor(log.finalReward)}`}>
                      {log.finalReward.toFixed(3)}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Desktop View - Table Layout */}
          <div className="hidden md:block">
            <div className="overflow-x-auto">
              <table className="w-full min-w-full">
                <thead className="bg-gray-50 dark:bg-gray-700 sticky top-0 z-10">
                  <tr>
                    <th className="px-3 lg:px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider min-w-[140px]">
                      Timestamp
                    </th>
                    <th className="px-3 lg:px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider min-w-[80px]">
                      Model
                    </th>
                    <th className="px-3 lg:px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider min-w-[200px]">
                      Original Prompt
                    </th>
                    <th className="px-3 lg:px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider min-w-[200px]">
                      Modified Prompt
                    </th>
                    <th className="px-3 lg:px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider min-w-[250px]">
                      AI Response
                    </th>
                    <th className="px-3 lg:px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider min-w-[160px]">
                      Feedback Summary
                    </th>
                    <th className="px-3 lg:px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider min-w-[100px]">
                      Final Reward
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200 dark:divide-gray-700 bg-white dark:bg-gray-800">
                  {sortedLogs.map((log) => (
                    <tr key={log.id} className="hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors">
                      <td className="px-3 lg:px-4 py-4 whitespace-nowrap">
                        <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
                          <Calendar size={14} />
                          <span className="text-xs">{formatTimestamp(log.timestamp)}</span>
                        </div>
                      </td>

                      <td className="px-3 lg:px-4 py-4 whitespace-nowrap">
                        <span
                          className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${getModelBadgeColor(log.model)}`}
                        >
                          <Brain size={12} />
                          {log.model}
                        </span>
                      </td>

                      <td className="px-3 lg:px-4 py-4 max-w-[200px]">
                        <div className="text-sm text-gray-900 dark:text-gray-100 break-words line-clamp-3">
                          {log.originalPrompt}
                        </div>
                      </td>

                      <td className="px-3 lg:px-4 py-4 max-w-[200px]">
                        <div className="text-sm text-gray-600 dark:text-gray-400 break-words line-clamp-3">
                          {log.modifiedPrompt}
                        </div>
                      </td>

                      <td className="px-3 lg:px-4 py-4 max-w-[250px]">
                        <div className="text-sm text-gray-900 dark:text-gray-100 break-words line-clamp-3">
                          {log.aiResponse}
                        </div>
                      </td>

                      <td className="px-3 lg:px-4 py-4">
                        <div className="text-xs space-y-1">
                          <div>Clarity: {log.feedbackSummary.promptClarity.toFixed(1)}/10</div>
                          <div>Consistency: {log.feedbackSummary.responseConsistency.toFixed(2)}</div>
                          <div>Sentiment: {log.feedbackSummary.sentimentScore.toFixed(2)}</div>
                          <div
                            className={
                              log.feedbackSummary.hallucinationFlag
                                ? "text-red-600 dark:text-red-400"
                                : "text-green-600 dark:text-green-400"
                            }
                          >
                            Hallucination: {log.feedbackSummary.hallucinationFlag ? "Yes" : "No"}
                          </div>
                        </div>
                      </td>

                      <td className="px-3 lg:px-4 py-4 whitespace-nowrap">
                        <div className={`text-lg font-semibold ${getRewardColor(log.finalReward)}`}>
                          {log.finalReward.toFixed(3)}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {sortedLogs.length === 0 && (
            <div className="text-center py-12">
              <div className="text-gray-500 dark:text-gray-400">
                {filterByModel === "all"
                  ? "No prompt logs available yet. Start chatting to see logs here."
                  : `No logs found for ${filterByModel} model. Try a different filter or start chatting.`}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
