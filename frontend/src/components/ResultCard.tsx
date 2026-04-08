import type { InferenceResult } from '../types'

function formatTs(seconds: number): string {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = seconds % 60
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${s.toFixed(2).padStart(5, '0')}`
  return `${String(m).padStart(2, '0')}:${s.toFixed(2).padStart(5, '0')}`
}

interface Props {
  result: InferenceResult
  index: number
}

export function ResultCard({ result, index }: Props) {
  return (
    <div className="group flex gap-3 rounded-lg border border-bg-border bg-bg-surface p-3 transition-colors hover:border-brand/30 hover:bg-bg-elevated">
      {/* Index */}
      <span className="mt-0.5 shrink-0 font-mono text-xs text-gray-600">
        #{String(index + 1).padStart(3, '0')}
      </span>

      {/* Body */}
      <div className="min-w-0 flex-1">
        <p className="text-sm leading-relaxed text-gray-100">{result.description}</p>
        <div className="mt-1.5 flex flex-wrap items-center gap-3 text-xs text-gray-500">
          <span className="font-mono text-emerald-500">{formatTs(result.timestamp)}</span>
          <span>{result.latency_ms.toFixed(0)} ms</span>
          <span>{result.token_count} tokens</span>
          <span className="font-mono text-gray-600">{result.model_id}</span>
        </div>
      </div>
    </div>
  )
}
