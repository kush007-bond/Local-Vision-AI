import { useState, useEffect, useRef, useCallback } from 'react'
import {
  ArrowLeft, Square, RefreshCw, ArrowDown, CheckCircle2,
  AlertCircle, Clock, Layers,
} from 'lucide-react'
import { api } from '../api'
import { useJobWebSocket } from '../hooks/useWebSocket'
import type { InferenceResult, Job, JobStatus, SourceType, WsMessage } from '../types'
import { StatusBadge } from './StatusBadge'
import { BackendBadge } from './BackendBadge'
import { ResultCard } from './ResultCard'
import { WebcamPreview } from './WebcamPreview'
import type { WebcamPreviewHandle } from './WebcamPreview'

interface Props {
  jobId: string
  onBack: () => void
  onJobUpdate: () => void
  sourceType?: SourceType
  deviceIndex?: number
  captureFps?: number
}

export function JobDetail({ jobId, onBack, onJobUpdate, sourceType, deviceIndex = 0, captureFps = 1 }: Props) {
  const [job, setJob]               = useState<Job | null>(null)
  const [results, setResults]       = useState<InferenceResult[]>([])
  const [status, setStatus]         = useState<JobStatus>('queued')
  const [autoScroll, setAutoScroll] = useState(true)
  const [stopping, setStopping]     = useState(false)
  const [error, setError]           = useState<string | null>(null)
  const scrollRef                   = useRef<HTMLDivElement>(null)
  const bottomRef                   = useRef<HTMLDivElement>(null)
  const webcamRef                   = useRef<WebcamPreviewHandle>(null)
  // Track timestamps we've already rendered to deduplicate WS replay on reconnect
  const seenTs                      = useRef<Set<number>>(new Set())

  // Load initial state
  useEffect(() => {
    api.getJob(jobId).then(j => {
      setJob(j)
      setResults(j.results)
      setStatus(j.status)
      if (j.status === 'failed' && j.error) setError(j.error)
      // Pre-populate seen set so WS replay doesn't create duplicates
      seenTs.current = new Set(j.results.map(r => r.timestamp))
    }).catch(() => setError('Failed to load job'))
  }, [jobId])

  // Auto-scroll
  useEffect(() => {
    if (autoScroll) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [results, autoScroll])

  const handleScroll = useCallback(() => {
    const el = scrollRef.current
    if (!el) return
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 60
    setAutoScroll(atBottom)
  }, [])

  // WebSocket streaming
  useJobWebSocket(jobId, {
    onMessage: useCallback((msg: WsMessage) => {
      if (msg.type === 'result') {
        // Deduplicate: server replays all buffered results on every new WS
        // connection (e.g. after slow inference pauses and client reconnects).
        // Skip results we already loaded from the REST fetch or a prior stream.
        if (seenTs.current.has(msg.data.timestamp)) return
        seenTs.current.add(msg.data.timestamp)
        setResults(prev => [...prev, msg.data])
      } else if (msg.type === 'status') {
        setStatus(msg.status)
        onJobUpdate()
      } else if (msg.type === 'complete') {
        setStatus(msg.status)
        onJobUpdate()
      } else if (msg.type === 'error') {
        setError(msg.message)
        setStatus('failed')
        onJobUpdate()
      }
    }, [onJobUpdate]),
  })

  const isRunning = status === 'running' || status === 'queued'

  // Frame capture loop for browser-driven webcam jobs.
  // Reads from the live <video> element via canvas and POSTs to /api/jobs/{id}/frame.
  useEffect(() => {
    if (sourceType !== 'webcam' || !isRunning) return

    const canvas = document.createElement('canvas')
    const startTime = Date.now()
    const intervalMs = Math.max(200, 1000 / captureFps)

    const id = setInterval(() => {
      const video = webcamRef.current?.getVideoElement()
      if (!video || video.readyState < 2 || video.videoWidth === 0) return

      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      ctx.drawImage(video, 0, 0)

      const base64 = canvas.toDataURL('image/jpeg', 0.8).split(',')[1]
      const timestamp = (Date.now() - startTime) / 1000

      api.pushWebcamFrame(jobId, base64, timestamp).catch(() => {
        // Silently drop — model may still be loading or job ended
      })
    }, intervalMs)

    return () => clearInterval(id)
  }, [sourceType, isRunning, jobId, captureFps])

  async function handleStop() {
    setStopping(true)
    try {
      await api.cancelJob(jobId)
      setStatus('cancelled')
      onJobUpdate()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Cancel failed')
    } finally {
      setStopping(false)
    }
  }

  // Elapsed time
  const elapsed = (() => {
    if (!job?.started_at) return null
    const start = new Date(job.started_at).getTime()
    const end = job.completed_at ? new Date(job.completed_at).getTime() : Date.now()
    const s = Math.floor((end - start) / 1000)
    return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${s % 60}s`
  })()

  const avgLatency = results.length
    ? (results.reduce((a, r) => a + r.latency_ms, 0) / results.length).toFixed(0)
    : null

  return (
    <div className="flex h-full flex-col">
      {/* ── Top bar ──────────────────────────────────────── */}
      <div className="flex shrink-0 items-center gap-3 border-b border-bg-border bg-bg-surface px-5 py-3">
        <button onClick={onBack} className="btn-ghost px-2 py-1 text-xs">
          <ArrowLeft className="h-4 w-4" /> Back
        </button>

        <div className="mx-1 h-4 w-px bg-bg-border" />

        <code className="font-mono text-xs text-gray-500">{jobId}</code>

        {job && <BackendBadge backend={job.backend} />}
        {job && <span className="text-sm font-medium text-gray-200">{job.model_id}</span>}

        <StatusBadge status={status} />

        <div className="flex-1" />

        {/* Stats chips */}
        <div className="flex items-center gap-3 text-xs text-gray-500">
          <span className="flex items-center gap-1">
            <Layers className="h-3.5 w-3.5" /> {results.length} frames
          </span>
          {elapsed && (
            <span className="flex items-center gap-1">
              <Clock className="h-3.5 w-3.5" /> {elapsed}
            </span>
          )}
          {avgLatency && (
            <span>avg {avgLatency} ms</span>
          )}
        </div>

        {isRunning && (
          <button
            className="btn-danger px-3 py-1.5 text-xs"
            onClick={() => void handleStop()}
            disabled={stopping}
          >
            {stopping
              ? <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-white/30 border-t-white" />
              : <Square className="h-3.5 w-3.5" />
            }
            Stop
          </button>
        )}
      </div>

      {/* ── Webcam live preview (only while job is active) ── */}
      {sourceType === 'webcam' && isRunning && (
        <div className="shrink-0 border-b border-bg-border bg-bg-surface px-5 py-3">
          <p className="mb-2 text-xs font-semibold uppercase tracking-widest text-gray-500">Camera Feed</p>
          <div className="mx-auto max-w-xs">
            <WebcamPreview ref={webcamRef} deviceIndex={deviceIndex} autoStart />
          </div>
        </div>
      )}

      {/* ── Results list ─────────────────────────────────── */}
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto p-5"
      >
        {error && (
          <div className="mb-4 flex items-center gap-2 rounded-lg border border-red-800 bg-red-950/50 px-4 py-3 text-sm text-red-400">
            <AlertCircle className="h-4 w-4 shrink-0" />
            {error}
          </div>
        )}

        {results.length === 0 && !error && (
          <div className="flex flex-col items-center justify-center gap-3 py-20 text-gray-600">
            {isRunning
              ? <><RefreshCw className="h-8 w-8 animate-spin opacity-40" /><p className="text-sm">Waiting for first frame…</p></>
              : <><CheckCircle2 className="h-8 w-8 opacity-40" /><p className="text-sm">No frames were processed.</p></>
            }
          </div>
        )}

        <div className="flex flex-col gap-2">
          {results.map((r, i) => (
            <ResultCard key={i} result={r} index={i} />
          ))}
        </div>

        <div ref={bottomRef} />
      </div>

      {/* ── Auto-scroll indicator ────────────────────────── */}
      {!autoScroll && isRunning && (
        <button
          onClick={() => {
            setAutoScroll(true)
            bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
          }}
          className="absolute bottom-6 right-6 flex items-center gap-1.5 rounded-full border border-bg-border bg-bg-elevated px-3 py-1.5 text-xs text-gray-400 shadow-lg hover:text-gray-100"
        >
          <ArrowDown className="h-3.5 w-3.5" /> Jump to latest
        </button>
      )}
    </div>
  )
}
