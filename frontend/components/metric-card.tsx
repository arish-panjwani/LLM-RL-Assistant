"use client"

import type { LucideIcon } from "lucide-react"

interface MetricCardProps {
  title: string
  value: number
  maxValue: number
  minValue?: number
  unit: string
  description: string
  icon: LucideIcon
  color: string
  isBoolean?: boolean
}

export default function MetricCard({
  title,
  value,
  maxValue,
  minValue = 0,
  unit,
  description,
  icon: Icon,
  color,
  isBoolean = false,
}: MetricCardProps) {
  const getColorClasses = (color: string) => {
    const colors = {
      blue: "bg-blue-100 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 border-blue-200 dark:border-blue-800",
      green:
        "bg-green-100 dark:bg-green-900/20 text-green-600 dark:text-green-400 border-green-200 dark:border-green-800",
      purple:
        "bg-purple-100 dark:bg-purple-900/20 text-purple-600 dark:text-purple-400 border-purple-200 dark:border-purple-800",
      pink: "bg-pink-100 dark:bg-pink-900/20 text-pink-600 dark:text-pink-400 border-pink-200 dark:border-pink-800",
      red: "bg-red-100 dark:bg-red-900/20 text-red-600 dark:text-red-400 border-red-200 dark:border-red-800",
      emerald:
        "bg-emerald-100 dark:bg-emerald-900/20 text-emerald-600 dark:text-emerald-400 border-emerald-200 dark:border-emerald-800",
    }
    return colors[color as keyof typeof colors] || colors.blue
  }

  const getProgressColor = (color: string) => {
    const colors = {
      blue: "bg-blue-500",
      green: "bg-green-500",
      purple: "bg-purple-500",
      pink: "bg-pink-500",
      red: "bg-red-500",
      emerald: "bg-emerald-500",
    }
    return colors[color as keyof typeof colors] || colors.blue
  }

  const normalizedValue = Math.max(0, Math.min(1, (value - minValue) / (maxValue - minValue)))
  const displayValue = isBoolean ? (value ? "True" : "False") : value.toFixed(2)

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4 hover:shadow-md transition-all">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className={`p-2 rounded-lg ${getColorClasses(color)}`}>
          <Icon size={20} />
        </div>
        <div className="text-right">
          <div className="text-2xl font-bold text-gray-800 dark:text-gray-200 transition-colors">
            {displayValue}
            {unit}
          </div>
        </div>
      </div>

      {/* Title */}
      <h3 className="font-medium text-gray-800 dark:text-gray-200 mb-2 transition-colors">{title}</h3>

      {/* Progress Bar (if not boolean) */}
      {!isBoolean && (
        <div className="mb-3">
          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 transition-colors">
            <div
              className={`h-2 rounded-full transition-all duration-300 ${getProgressColor(color)}`}
              style={{ width: `${normalizedValue * 100}%` }}
            />
          </div>
        </div>
      )}

      {/* Description */}
      <p className="text-sm text-gray-600 dark:text-gray-400 transition-colors">{description}</p>
    </div>
  )
}
