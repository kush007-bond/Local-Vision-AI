import { useEffect, useRef, useCallback } from 'react'
import { createJobWebSocket } from '../api'
import type { WsMessage } from '../types'

interface Options {
  onMessage: (msg: WsMessage) => void
  onOpen?: () => void
  onClose?: () => void
}

export function useJobWebSocket(jobId: string | null, opts: Options) {
  const wsRef = useRef<WebSocket | null>(null)
  const optsRef = useRef(opts)
  optsRef.current = opts

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.onmessage = null
      wsRef.current.onopen = null
      wsRef.current.onclose = null
      wsRef.current.onerror = null
      wsRef.current.close()
      wsRef.current = null
    }
  }, [])

  useEffect(() => {
    if (!jobId) return
    disconnect()

    const ws = createJobWebSocket(jobId)
    wsRef.current = ws

    ws.onopen = () => optsRef.current.onOpen?.()

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data as string) as WsMessage
        // Silently drop server keepalive pings — they carry no data
        if (msg.type === 'ping') return
        optsRef.current.onMessage(msg)
      } catch {
        // ignore parse errors
      }
    }

    ws.onclose = () => optsRef.current.onClose?.()
    // On network error, close cleanly (browser fires onclose after onerror anyway)
    ws.onerror = () => ws.close()

    return disconnect
  }, [jobId, disconnect])

  return { disconnect }
}
