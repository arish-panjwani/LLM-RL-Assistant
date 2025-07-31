"use client"

import { useState } from "react"
import MetricCard from "@/components/metric-card"
import { Brain, Target, Palette, Heart, AlertTriangle, CheckCircle } from "lucide-react"

export interface AIMetrics {
  promptClarity: number // 1-10 scale
  responseConsistency: number // 0-1 (cosine similarity)
  lexicalDiversity: number // 0-1 scale
  sentimentScore: number // -1 to 1
  hallucinationFlag: boolean
  factAccuracy: number // 0-1 scale
}

export default function FeedbackTab() {
  // Mock data - TODO: Replace with real metrics from Flask API
  const [metrics] = useState<AIMetrics>({
    promptClarity: 8.5,
    responseConsistency: 0.87,
    lexicalDiversity: 0.73,
    sentimentScore: 0.65,
    hallucinationFlag: false,
    factAccuracy: 0.92,
  })

  const metricCards = [
    {
      title: "Prompt Clarity",
      value: metrics.promptClarity,
      maxValue: 10,
      unit: "/10",
      description: "How clear and well-structured the user prompts are",
      icon: Target,
      color: "blue",
    },
    {
      title: "Response Consistency",
      value: metrics.responseConsistency,
      maxValue: 1,
      unit: "",
      description: "Cosine similarity between related responses",
      icon: Brain,
      color: "green",
    },
    {
      title: "Lexical Diversity",
      value: metrics.lexicalDiversity,
      maxValue: 1,
      unit: "",
      description: "Variety of vocabulary used in responses",
      icon: Palette,
      color: "purple",
    },
    {
      title: "Sentiment Score",
      value: metrics.sentimentScore,
      maxValue: 1,
      minValue: -1,
      unit: "",
      description: "Overall emotional tone of the conversation",
      icon: Heart,
      color: "pink",
    },
    {
      title: "Hallucination Flag",
      value: metrics.hallucinationFlag ? 1 : 0,
      maxValue: 1,
      unit: "",
      description: "Detection of potentially fabricated information",
      icon: AlertTriangle,
      color: metrics.hallucinationFlag ? "red" : "green",
      isBoolean: true,
    },
    {
      title: "Fact Accuracy",
      value: metrics.factAccuracy,
      maxValue: 1,
      unit: "",
      description: "Accuracy of factual claims in responses",
      icon: CheckCircle,
      color: "emerald",
    },
  ]

  return (
    <div className="p-4">
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-2 transition-colors">
          AI Performance Metrics
        </h2>
        <p className="text-gray-600 dark:text-gray-400 text-sm transition-colors">
          Real-time analytics of AI assistant performance and interaction quality
        </p>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
        {metricCards.map((metric, index) => (
          <MetricCard key={index} {...metric} />
        ))}
      </div>

      {/* Additional Info */}
      <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4 transition-colors">
        <h3 className="font-medium text-blue-900 dark:text-blue-100 mb-2 transition-colors">About These Metrics</h3>
        <ul className="text-sm text-blue-800 dark:text-blue-200 space-y-1 transition-colors">
          <li>• Metrics are calculated in real-time based on conversation data</li>
          <li>• Higher scores generally indicate better AI performance</li>
          <li>• Hallucination flag alerts when AI may be generating false information</li>
          <li>• Sentiment ranges from -1 (negative) to +1 (positive)</li>
        </ul>
      </div>
    </div>
  )
}
