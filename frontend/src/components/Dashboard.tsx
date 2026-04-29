import { useEffect, useState, useCallback } from 'react'
import { Play, RefreshCw, Inbox, Clock, CheckCircle, XCircle } from 'lucide-react'
import { api } from '../api'
import type { JobSummary, View } from '../types'
import { StatusBadge } from './StatusBadge'
import { BackendBadge } from './BackendBadge'

interface Props {
  onNavigate: (view: View, jobId?: string) => void
  refreshTick: number
}

function StatCard({ label, value, icon: Icon, color }: {
  label: string; value: number; icon: React.ElementType; color: string
}) {
  return (
    <div className="card flex items-center gap-4">
      <div className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-lg ${color}`}>
        <Icon className="h-5 w-5" />
      </div>
      <div>
        <p className="text-2xl font-bold text-gray-100">{value}</p>
        <p className="text-xs text-gray-500">{label}</p>
      </div>
    </div>
  )
}

function elapsedLabel(startedAt: string, completedAt?: string): string {
  const start = new Date(startedAt).getTime()
  const end = completedAt ? new Date(completedAt).getTime() : Date.now()
  const secs = Math.floor((end - start) / 1000)
  if (secs < 60) return `${secs}s`
  const m = Math.floor(secs / 60)
  const s = secs % 60
  return `${m}m ${s}s`
}

export function Dashboard({ onNavigate, refreshTick }: Props) {
  const [jobs, setJobs] = useState<JobSummary[]>([])
  const [loading, setLoading] = useState(true)

  const load = useCallback(async () => {
    try {
      const list = await api.listJobs()
      setJobs(list)
    } catch {
      // server may be offline
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { void load() }, [load, refreshTick])

  const active    = jobs.filter(j => j.status === 'running' || j.status === 'queued')
  const completed = jobs.filter(j => j.status === 'completed')
  const failed    = jobs.filter(j => j.status === 'failed' || j.status === 'cancelled')

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center text-gray-500">
        <RefreshCw className="h-5 w-5 animate-spin" />
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-6 p-4 sm:p-6">
      {/* Stats */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <StatCard label="Active jobs"     value={active.length}    icon={Play}         color="bg-emerald-950 text-emerald-400" />
        <StatCard label="Completed"       value={completed.length} icon={CheckCircle}  color="bg-blue-950 text-blue-400" />
        <StatCard label="Failed / cancelled" value={failed.length} icon={XCircle}      color="bg-red-950 text-red-400" />
      </div>

      {/* Jobs list */}
      <div>
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-semibold text-gray-300">All Jobs</h2>
          <button
            onClick={() => void load()}
            className="btn-ghost px-2 py-1 text-xs"
          >
            <RefreshCw className="h-3.5 w-3.5" /> Refresh
          </button>
        </div>

        {jobs.length === 0 ? (
          <div className="flex flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-bg-border py-16 text-gray-500">
            <Inbox className="h-10 w-10 opacity-40" />
            <p className="text-sm">No jobs yet</p>
            <button className="btn-primary" onClick={() => onNavigate('new-job')}>
              <Play className="h-4 w-4" /> Start a pipeline
            </button>
          </div>
        ) : (
          <div className="flex flex-col gap-2">
            {jobs.map(job => (
              <button
                key={job.job_id}
                onClick={() => onNavigate('job-detail', job.job_id)}
                className="card flex w-full items-center gap-4 text-left transition-colors hover:border-brand/30 hover:bg-bg-elevated"
              >
                {/* Status dot */}
                <StatusBadge status={job.status} />

                {/* Info */}
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <BackendBadge backend={job.backend} />
                    <span className="truncate text-sm font-medium text-gray-200">
                      {job.model_id}
                    </span>
                  </div>
                  <p className="mt-0.5 truncate text-xs text-gray-500">{job.source}</p>
                </div>

                {/* Stats */}
                <div className="flex shrink-0 flex-col items-end gap-0.5 text-xs text-gray-500">
                  <span className="font-medium text-gray-300">{job.result_count} frames</span>
                  <span className="flex items-center gap-1">
                    <Clock className="h-3 w-3" />
                    {elapsedLabel(job.started_at, job.completed_at)}
                  </span>
                </div>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
