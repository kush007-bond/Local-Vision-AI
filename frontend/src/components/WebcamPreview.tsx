import { useState, useEffect, useRef, useCallback, forwardRef, useImperativeHandle } from 'react'
import { Camera, CameraOff, FlipHorizontal2, ShieldAlert, RefreshCw } from 'lucide-react'

export interface WebcamPreviewHandle {
  getVideoElement: () => HTMLVideoElement | null
}

interface Props {
  /** Called whenever the active camera changes, with the index into the enumerated device list. */
  onDeviceChange?: (deviceIndex: number, deviceId: string) => void
  /** Pre-selected device index (from parent state). */
  deviceIndex?: number
  autoStart?: boolean
}

type FacingMode = 'user' | 'environment'

function isSecureContext() {
  return window.isSecureContext || location.hostname === 'localhost' || location.hostname === '127.0.0.1'
}

export const WebcamPreview = forwardRef<WebcamPreviewHandle, Props>(
  function WebcamPreview({ deviceIndex = 0, autoStart = false, onDeviceChange }, ref) {
    const videoRef  = useRef<HTMLVideoElement>(null)
    const streamRef = useRef<MediaStream | null>(null)

    const [active, setActive]         = useState(false)
    const [error, setError]           = useState<string | null>(null)
    const [loading, setLoading]       = useState(false)
    // Only populated after permission is granted
    const [cameras, setCameras]       = useState<MediaDeviceInfo[]>([])
    const [activeCamId, setActiveCamId] = useState<string | null>(null)

    const stopStream = useCallback(() => {
      streamRef.current?.getTracks().forEach(t => t.stop())
      streamRef.current = null
      if (videoRef.current) videoRef.current.srcObject = null
    }, [])

    /** Enumerate cameras — only returns labelled devices (requires prior permission). */
    const enumerateCameras = useCallback(async (): Promise<MediaDeviceInfo[]> => {
      try {
        const all = await navigator.mediaDevices.enumerateDevices()
        // Filter to video inputs that have a real deviceId (not empty string)
        const cams = all.filter(d => d.kind === 'videoinput' && d.deviceId !== '')
        setCameras(cams)
        return cams
      } catch {
        return []
      }
    }, [])

    const applyStream = useCallback(async (stream: MediaStream, camId: string, cams: MediaDeviceInfo[]) => {
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play().catch(() => {})
      }
      setActiveCamId(camId)
      setActive(true)
      // Notify parent with the index in the enumerated list
      const idx = cams.findIndex(c => c.deviceId === camId)
      onDeviceChange?.(idx >= 0 ? idx : 0, camId)
    }, [onDeviceChange])

    /** Start a stream for a specific deviceId. */
    const startById = useCallback(async (deviceId: string) => {
      if (!isSecureContext()) {
        setError('Camera requires HTTPS. Use the Cloudflare tunnel URL on remote devices.')
        return
      }
      if (!navigator.mediaDevices?.getUserMedia) {
        setError('Camera API not available in this browser.')
        return
      }
      stopStream()
      setError(null)
      setLoading(true)
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { deviceId: { exact: deviceId } },
          audio: false,
        })
        const cams = await enumerateCameras()
        await applyStream(stream, deviceId, cams)
      } catch (e) {
        setError((e as DOMException)?.message ?? 'Could not access camera')
      } finally {
        setLoading(false)
      }
    }, [stopStream, enumerateCameras, applyStream])

    /** Initial start — uses facingMode so it works on mobile before we have deviceIds. */
    const startInitial = useCallback(async (facing: FacingMode = 'user') => {
      if (!isSecureContext()) {
        setError('Camera requires HTTPS. Use the Cloudflare tunnel URL on remote devices.')
        return
      }
      if (!navigator.mediaDevices?.getUserMedia) {
        setError('Camera API not available in this browser.')
        return
      }
      stopStream()
      setError(null)
      setLoading(true)
      try {
        // First pass: get any stream so the browser grants permission
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: { ideal: facing } },
          audio: false,
        })
        // Now enumerate — labels are populated after permission is granted
        const cams = await enumerateCameras()

        // Find the actual deviceId of the track that was granted
        const trackSettings = stream.getVideoTracks()[0]?.getSettings()
        const grantedId = trackSettings?.deviceId ?? ''

        // Pick the right camera: prefer the one matching granted deviceId, else use deviceIndex
        const target = cams.find(c => c.deviceId === grantedId)
          ?? cams[Math.min(deviceIndex, cams.length - 1)]

        if (target && target.deviceId !== grantedId) {
          // The granted stream is the wrong camera; restart with the right one
          stream.getTracks().forEach(t => t.stop())
          const correctStream = await navigator.mediaDevices.getUserMedia({
            video: { deviceId: { exact: target.deviceId } },
            audio: false,
          })
          await applyStream(correctStream, target.deviceId, cams)
        } else {
          await applyStream(stream, grantedId || (target?.deviceId ?? ''), cams)
        }
      } catch (e) {
        const err = e as DOMException
        if (err?.name === 'NotAllowedError') {
          setError('Camera permission denied. Allow camera access in your browser settings.')
        } else {
          // Retry with no constraints as last resort
          try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
            const cams = await enumerateCameras()
            const id = stream.getVideoTracks()[0]?.getSettings()?.deviceId ?? ''
            await applyStream(stream, id, cams)
          } catch (e2) {
            setError((e2 as DOMException)?.message ?? 'Could not access camera')
          }
        }
      } finally {
        setLoading(false)
      }
    }, [stopStream, enumerateCameras, applyStream, deviceIndex])

    const stop = useCallback(() => {
      stopStream()
      setActive(false)
      setActiveCamId(null)
    }, [stopStream])

    /** Flip between front and back by toggling to the next camera in the list. */
    const flip = useCallback(() => {
      if (cameras.length < 2) return
      const currentIdx = cameras.findIndex(c => c.deviceId === activeCamId)
      const nextIdx = (currentIdx + 1) % cameras.length
      void startById(cameras[nextIdx].deviceId)
    }, [cameras, activeCamId, startById])

    useImperativeHandle(ref, () => ({ getVideoElement: () => videoRef.current }), [])

    useEffect(() => {
      if (autoStart) void startInitial('user')
      return () => stopStream()
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [autoStart])

    const insecure = !isSecureContext()

    if (insecure) {
      return (
        <div className="flex items-start gap-2 rounded-lg border border-amber-800 bg-amber-950/40 px-3 py-3 text-xs text-amber-400">
          <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0" />
          <div>
            <p className="font-semibold">Camera blocked — insecure connection</p>
            <p className="mt-0.5 text-amber-500">
              Open this app via the <strong>Cloudflare tunnel</strong> HTTPS URL instead of the local IP address.
            </p>
          </div>
        </div>
      )
    }

    return (
      <div className="flex flex-col gap-2">
        {/* Video preview */}
        <div className="relative overflow-hidden rounded-lg border border-bg-border bg-black" style={{ aspectRatio: '16/9' }}>
          <video ref={videoRef} className="h-full w-full object-contain" muted playsInline />

          {!active && !loading && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-gray-500">
              <CameraOff className="h-8 w-8" />
              <span className="text-xs">Preview off</span>
            </div>
          )}

          {loading && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/60">
              <RefreshCw className="h-6 w-6 animate-spin text-gray-400" />
            </div>
          )}

          {active && (
            <>
              {autoStart && (
                <div className="absolute top-2 left-2 flex items-center gap-1 rounded-full bg-red-600/90 px-2 py-0.5">
                  <span className="h-1.5 w-1.5 rounded-full bg-white animate-pulse" />
                  <span className="text-xs font-medium text-white">Live</span>
                </div>
              )}

              {cameras.length > 1 && (
                <button
                  type="button"
                  onClick={flip}
                  title="Switch camera"
                  className="absolute top-2 right-2 flex items-center justify-center rounded-full bg-black/60 p-2 text-white backdrop-blur-sm hover:bg-black/80 transition-colors"
                >
                  <FlipHorizontal2 className="h-4 w-4" />
                </button>
              )}
            </>
          )}
        </div>

        {error && <p className="text-xs text-red-400">{error}</p>}

        {/* Camera picker — only shown after permission granted (labels populated) */}
        {cameras.length > 1 && cameras[0]?.label && (
          <select
            className="select text-xs py-1.5"
            value={activeCamId ?? ''}
            onChange={e => void startById(e.target.value)}
          >
            {cameras.map((d, i) => (
              <option key={d.deviceId} value={d.deviceId}>
                {d.label || `Camera ${i + 1}`}
              </option>
            ))}
          </select>
        )}

        {!autoStart && (
          <button
            type="button"
            onClick={active ? stop : () => void startInitial('user')}
            className={`flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors ${
              active
                ? 'border-red-800 bg-red-950/40 text-red-400 hover:bg-red-950/70'
                : 'border-bg-border bg-bg-elevated text-gray-400 hover:text-gray-200'
            }`}
          >
            {active
              ? <><CameraOff className="h-3.5 w-3.5" /> Stop preview</>
              : <><Camera className="h-3.5 w-3.5" /> Start preview</>
            }
          </button>
        )}
      </div>
    )
  }
)
