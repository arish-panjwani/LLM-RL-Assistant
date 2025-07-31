// Audio helper for tick sounds and feedback audio
export class AudioHelper {
  private static tickSound: HTMLAudioElement | null = null
  private static soundEnabled = true

  // Initialize tick sound
  static initializeTickSound() {
    if (typeof window !== "undefined" && !this.tickSound) {
      this.tickSound = new Audio()
      this.createTickSound()
    }
  }

  // Create a simple tick sound programmatically
  private static createTickSound() {
    if (typeof window === "undefined" || !this.tickSound) return

    try {
      // Create a simple beep sound using Web Audio API
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)()

      // Create a short beep sound
      const duration = 0.1
      const sampleRate = audioContext.sampleRate
      const numSamples = duration * sampleRate
      const buffer = audioContext.createBuffer(1, numSamples, sampleRate)
      const channelData = buffer.getChannelData(0)

      // Generate a pleasant tick sound (short sine wave with envelope)
      for (let i = 0; i < numSamples; i++) {
        const t = i / sampleRate
        const envelope = Math.exp(-t * 10) // Exponential decay
        const frequency = 800 // Pleasant tick frequency
        channelData[i] = Math.sin(2 * Math.PI * frequency * t) * envelope * 0.3
      }

      // Convert to data URL and set as audio source
      this.bufferToDataURL(buffer)
        .then((dataURL) => {
          if (this.tickSound) {
            this.tickSound.src = dataURL
            this.tickSound.volume = 0.4
          }
        })
        .catch(() => {
          // Fallback to a simple beep sound
          this.createSimpleBeep()
        })
    } catch (error) {
      // Fallback to a simple data URL beep
      if (this.tickSound) {
        this.createSimpleBeep()
      }
    }
  }

  // Convert audio buffer to data URL
  private static async bufferToDataURL(buffer: AudioBuffer): Promise<string> {
    return new Promise((resolve, reject) => {
      try {
        // Create a simple WAV file from the buffer
        const length = buffer.length
        const arrayBuffer = new ArrayBuffer(44 + length * 2)
        const view = new DataView(arrayBuffer)

        // WAV header
        const writeString = (offset: number, string: string) => {
          for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i))
          }
        }

        writeString(0, "RIFF")
        view.setUint32(4, 36 + length * 2, true)
        writeString(8, "WAVE")
        writeString(12, "fmt ")
        view.setUint32(16, 16, true)
        view.setUint16(20, 1, true)
        view.setUint16(22, 1, true)
        view.setUint32(24, buffer.sampleRate, true)
        view.setUint32(28, buffer.sampleRate * 2, true)
        view.setUint16(32, 2, true)
        view.setUint16(34, 16, true)
        writeString(36, "data")
        view.setUint32(40, length * 2, true)

        // Convert float samples to 16-bit PCM
        const channelData = buffer.getChannelData(0)
        let offset = 44
        for (let i = 0; i < length; i++) {
          const sample = Math.max(-1, Math.min(1, channelData[i]))
          view.setInt16(offset, sample * 0x7fff, true)
          offset += 2
        }

        const blob = new Blob([arrayBuffer], { type: "audio/wav" })
        const reader = new FileReader()
        reader.onload = () => resolve(reader.result as string)
        reader.onerror = reject
        reader.readAsDataURL(blob)
      } catch (error) {
        reject(error)
      }
    })
  }

  // Create a simple beep sound as fallback
  private static createSimpleBeep() {
    if (!this.tickSound) return

    // Create a simple beep using oscillator (fallback)
    try {
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)()
      const oscillator = audioContext.createOscillator()
      const gainNode = audioContext.createGain()

      oscillator.connect(gainNode)
      gainNode.connect(audioContext.destination)

      oscillator.frequency.setValueAtTime(800, audioContext.currentTime)
      gainNode.gain.setValueAtTime(0.1, audioContext.currentTime)
      gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.1)

      oscillator.start(audioContext.currentTime)
      oscillator.stop(audioContext.currentTime + 0.1)
    } catch (error) {
      console.warn("Could not create fallback beep sound:", error)
    }
  }

  // Play the tick sound
  static playTickSound() {
    if (!this.soundEnabled || !this.tickSound) return

    try {
      // Reset and play the sound
      this.tickSound.currentTime = 0
      this.tickSound.play().catch(() => {
        // If audio file fails, try the simple beep fallback
        this.createSimpleBeep()
      })
    } catch (error) {
      console.warn("Could not play tick sound:", error)
      this.createSimpleBeep()
    }
  }

  // Enable/disable sound
  static setSoundEnabled(enabled: boolean) {
    this.soundEnabled = enabled
  }

  // Check if sound is enabled
  static isSoundEnabled(): boolean {
    return this.soundEnabled
  }
}
