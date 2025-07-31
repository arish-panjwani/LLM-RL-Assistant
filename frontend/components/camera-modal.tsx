"use client"

import { useState, useRef, useEffect } from "react"
import { X, Camera, RotateCcw, Upload } from "lucide-react"

interface CameraModalProps {
  isOpen: boolean
  onClose: () => void
  onCapture: (file: File) => void
  onUpload: () => void
}

export default function CameraModal({ isOpen, onClose, onCapture, onUpload }: CameraModalProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [facingMode, setFacingMode] = useState<"user" | "environment">("environment")
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [hasMultipleCameras, setHasMultipleCameras] = useState(false)

  // Check for multiple cameras
  useEffect(() => {
    if (isOpen) {
      navigator.mediaDevices
        .enumerateDevices()
        .then((devices) => {
          const videoDevices = devices.filter((device) => device.kind === "videoinput")
          setHasMultipleCameras(videoDevices.length > 1)
        })
        .catch(() => setHasMultipleCameras(false))
    }
  }, [isOpen])

  // Start camera when modal opens
  useEffect(() => {
    if (isOpen) {
      startCamera()
    } else {
      stopCamera()
    }

    return () => stopCamera()
  }, [isOpen, facingMode])

  const startCamera = async () => {
    setIsLoading(true)
    setError(null)

    try {
      // Stop existing stream
      if (stream) {
        stream.getTracks().forEach((track) => track.stop())
      }

      const constraints = {
        video: {
          facingMode: facingMode,
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      }

      const newStream = await navigator.mediaDevices.getUserMedia(constraints)
      setStream(newStream)

      if (videoRef.current) {
        videoRef.current.srcObject = newStream
        videoRef.current.play()
      }
    } catch (err) {
      console.error("Camera access error:", err)
      setError("Unable to access camera. Please check permissions.")
    } finally {
      setIsLoading(false)
    }
  }

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop())
      setStream(null)
    }
  }

  const switchCamera = () => {
    setFacingMode((prev) => (prev === "user" ? "environment" : "user"))
  }

  const capturePhoto = () => {
    if (!videoRef.current || !canvasRef.current) return

    const video = videoRef.current
    const canvas = canvasRef.current
    const ctx = canvas.getContext("2d")

    if (!ctx) return

    // Set canvas dimensions to match video
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight

    // Draw the current video frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

    // Convert to blob and create file
    canvas.toBlob(
      (blob) => {
        if (blob) {
          const timestamp = new Date().toISOString().replace(/[:.]/g, "-")
          const file = new File([blob], `camera-capture-${timestamp}.jpg`, {
            type: "image/jpeg",
          })
          onCapture(file)
          onClose()
        }
      },
      "image/jpeg",
      0.9,
    )
  }

  const handleUpload = () => {
    onUpload()
    onClose()
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black bg-opacity-90 flex items-center justify-center z-50">
      <div className="relative w-full h-full max-w-2xl max-h-[90vh] bg-black rounded-lg overflow-hidden">
        {/* Header */}
        <div className="absolute top-0 left-0 right-0 z-10 flex items-center justify-between p-4 bg-gradient-to-b from-black/70 to-transparent">
          <h3 className="text-white font-semibold">Camera</h3>
          <button
            onClick={onClose}
            className="p-2 text-white hover:bg-white/20 rounded-full transition-colors touch-manipulation"
          >
            <X size={24} />
          </button>
        </div>

        {/* Camera Feed */}
        <div className="relative w-full h-full flex items-center justify-center">
          {isLoading && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/50 z-10">
              <div className="text-white text-center">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white mx-auto mb-2"></div>
                <p>Starting camera...</p>
              </div>
            </div>
          )}

          {error && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/80 z-10">
              <div className="text-white text-center p-4">
                <p className="mb-4">{error}</p>
                <button
                  onClick={handleUpload}
                  className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
                >
                  Choose from Gallery
                </button>
              </div>
            </div>
          )}

          <video
            ref={videoRef}
            className="w-full h-full object-cover"
            playsInline
            muted
            style={{ transform: facingMode === "user" ? "scaleX(-1)" : "none" }}
          />

          {/* Hidden canvas for capture */}
          <canvas ref={canvasRef} className="hidden" />
        </div>

        {/* Controls */}
        <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/70 to-transparent">
          <div className="flex items-center justify-center gap-4">
            {/* Upload Button */}
            <button
              onClick={handleUpload}
              className="p-3 bg-gray-600/80 text-white rounded-full hover:bg-gray-500/80 transition-colors touch-manipulation"
              title="Upload from Gallery"
            >
              <Upload size={24} />
            </button>

            {/* Capture Button */}
            <button
              onClick={capturePhoto}
              disabled={isLoading || !!error}
              className="p-4 bg-white text-black rounded-full hover:bg-gray-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed touch-manipulation shadow-lg"
              title="Take Photo"
            >
              <Camera size={32} />
            </button>

            {/* Switch Camera Button (only show if multiple cameras available) */}
            {hasMultipleCameras && (
              <button
                onClick={switchCamera}
                disabled={isLoading}
                className="p-3 bg-gray-600/80 text-white rounded-full hover:bg-gray-500/80 transition-colors disabled:opacity-50 touch-manipulation"
                title={`Switch to ${facingMode === "user" ? "Back" : "Front"} Camera`}
              >
                <RotateCcw size={24} />
              </button>
            )}
          </div>

          {/* Camera Mode Indicator */}
          {hasMultipleCameras && (
            <div className="text-center mt-2">
              <span className="text-white/80 text-sm">{facingMode === "user" ? "Front Camera" : "Back Camera"}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
