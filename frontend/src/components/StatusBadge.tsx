import type { JobStatus } from '../types'

const CONFIG: Record<JobStatus, { label: string; classes: string; dot: string }> = {
  queued:    { label: 'Queued',    classes: 'bg-zinc-800 text-zinc-400',         dot: 'bg-zinc-500' },
  running:   { label: 'Running',   classes: 'bg-emerald-950 text-emerald-400',   dot: 'bg-emerald-400 animate-pulse' },
  completed: { label: 'Completed', classes: 'bg-blue-950 text-blue-400',         dot: 'bg-blue-400' },
  failed:    { label: 'Failed',    classes: 'bg-red-950 text-red-400',           dot: 'bg-red-400' },
  cancelled: { label: 'Cancelled', classes: 'bg-amber-950 text-amber-400',       dot: 'bg-amber-400' },
}

export function StatusBadge({ status }: { status: JobStatus }) {
  const { label, classes, dot } = CONFIG[status] ?? CONFIG.queued
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium ${classes}`}>
      <span className={`h-1.5 w-1.5 rounded-full ${dot}`} />
      {label}
    </span>
  )
}
