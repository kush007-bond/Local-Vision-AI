import { useState, useEffect, useRef, useCallback } from 'react'
import { Camera, CameraOff } from 'lucide-react'

interface Props {
  deviceIndex: number
  /** If true, camera starts automatically on mount and the toggle button is hidden. */
  autoStart?: boolean
}

export function WebcamPreview({ deviceIndex, autoStart = false }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const [active, setActive] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const start = useCallback(async () => {
    setError(null)
    try {
      const constraints: MediaStreamConstraints = {
        video: { deviceId: deviceIndex > 0 ? { exact: String(deviceIndex) } : undefined },
      }
      const stream = await navigator.mediaDevices.getUserMedia(constraints)
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        videoRef.current.play().catch(() => {/* autoplay may be blocked */})
      }
      setActive(true)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not access camera')
    }
  }, [deviceIndex])

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach(t => t.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    setActive(false)
  }, [])

  // Auto-start on mount when requested
  useEffect(() => {
    if (autoStart) void start()
  }, [autoStart, start])

  // Stop stream on unmount
  useEffect(() => () => { stop() }, [stop])

  return (
    <div className="flex flex-col gap-2">
      <div className="relative overflow-hidden rounded-lg border border-bg-border bg-black" style={{ aspectRatio: '16/9' }}>
        <video
          ref={videoRef}
          className="h-full w-full object-contain"
          muted
          playsInline
        />
        {!active && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-gray-500">
            <CameraOff className="h-8 w-8" />
            <span className="text-xs">Preview off</span>
          </div>
        )}
        {autoStart && active && (
          <div className="absolute top-2 left-2 flex items-center gap-1 rounded-full bg-red-600/90 px-2 py-0.5">
            <span className="h-1.5 w-1.5 rounded-full bg-white animate-pulse" />
            <span className="text-xs font-medium text-white">Live</span>
          </div>
        )}
      </div>
      {error && <p className="text-xs text-red-400">{error}</p>}
      {!autoStart && (
        <button
          type="button"
          onClick={active ? stop : () => void start()}
          className={`flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors ${
            active
              ? 'border-red-800 bg-red-950/40 text-red-400 hover:bg-red-950/70'
              : 'border-bg-border bg-bg-elevated text-gray-400 hover:text-gray-200'
          }`}
        >
          {active ? <><CameraOff className="h-3.5 w-3.5" /> Stop preview</> : <><Camera className="h-3.5 w-3.5" /> Start preview</>}
        </button>
      )}
    </div>
  )
}
