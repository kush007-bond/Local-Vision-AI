import type { BackendInfo, Job, JobConfig, JobSummary } from './types'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init)
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(body.detail ?? res.statusText)
  }
  return res.json() as Promise<T>
}

export const api = {
  health: () => request<{ status: string; version: string }>('/health'),

  listBackends: () =>
    request<{ backends: BackendInfo[] }>('/api/backends').then(r => r.backends),

  createJob: (config: JobConfig) =>
    request<JobSummary>('/api/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    }),

  listJobs: () =>
    request<{ jobs: JobSummary[] }>('/api/jobs').then(r => r.jobs),

  getJob: (jobId: string) =>
    request<Job>(`/api/jobs/${jobId}`),

  cancelJob: (jobId: string) =>
    request<{ job_id: string; status: string }>(`/api/jobs/${jobId}`, {
      method: 'DELETE',
    }),

  pushWebcamFrame: (jobId: string, imageBase64: string, timestamp: number) =>
    request<{ ok: boolean }>(`/api/jobs/${jobId}/frame`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageBase64, timestamp }),
    }),
}

export function createJobWebSocket(jobId: string): WebSocket {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return new WebSocket(`${proto}//${window.location.host}/ws/${jobId}`)
}
