import { useState, useEffect, useCallback } from 'react'
import { api } from './api'
import type { SourceType, View } from './types'
import { Header } from './components/Header'
import { Sidebar } from './components/Sidebar'
import { Dashboard } from './components/Dashboard'
import { NewJobForm } from './components/NewJobForm'
import { JobDetail } from './components/JobDetail'

export default function App() {
  const [view, setView]             = useState<View>('dashboard')
  const [activeJobId, setActiveJobId] = useState<string | null>(null)
  const [activeJobSource, setActiveJobSource] = useState<{ type: SourceType; deviceIndex: number } | null>(null)
  const [serverOnline, setServerOnline] = useState(false)
  const [activeCount, setActiveCount]   = useState(0)
  const [dashRefresh, setDashRefresh]   = useState(0)

  // Poll server health every 5 s
  useEffect(() => {
    let cancelled = false
    const check = async () => {
      try {
        await api.health()
        if (!cancelled) setServerOnline(true)
      } catch {
        if (!cancelled) setServerOnline(false)
      }
    }
    void check()
    const id = setInterval(() => void check(), 5000)
    return () => { cancelled = true; clearInterval(id) }
  }, [])

  // Keep active job count up to date
  useEffect(() => {
    const tick = async () => {
      try {
        const jobs = await api.listJobs()
        setActiveCount(jobs.filter(j => j.status === 'running' || j.status === 'queued').length)
      } catch { /* offline */ }
    }
    void tick()
    const id = setInterval(() => void tick(), 3000)
    return () => clearInterval(id)
  }, [dashRefresh])

  const handleJobCreated = useCallback((jobId: string, sourceType: SourceType, deviceIndex: number) => {
    setActiveJobId(jobId)
    setActiveJobSource({ type: sourceType, deviceIndex })
    setView('job-detail')
    setDashRefresh(n => n + 1)
  }, [])

  const handleNavigate = useCallback((v: View, jobId?: string) => {
    if (v === 'job-detail' && jobId) setActiveJobId(jobId)
    setView(v)
  }, [])

  const handleJobUpdate = useCallback(() => {
    setDashRefresh(n => n + 1)
  }, [])

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-bg-base">
      <Header serverOnline={serverOnline} />

      <div className="flex flex-1 overflow-hidden">
        <Sidebar
          current={view}
          onChange={v => setView(v)}
          activeCount={activeCount}
        />

        <main className="relative flex-1 overflow-y-auto">
          {view === 'dashboard' && (
            <Dashboard
              onNavigate={handleNavigate}
              refreshTick={dashRefresh}
            />
          )}

          {view === 'new-job' && (
            <NewJobForm onJobCreated={handleJobCreated} />
          )}

          {view === 'job-detail' && activeJobId && (
            <JobDetail
              jobId={activeJobId}
              onBack={() => setView('dashboard')}
              onJobUpdate={handleJobUpdate}
              sourceType={activeJobSource?.type}
              deviceIndex={activeJobSource?.deviceIndex}
            />
          )}

          {view === 'job-detail' && !activeJobId && (
            <div className="flex h-full items-center justify-center text-gray-500 text-sm">
              No job selected.
            </div>
          )}
        </main>
      </div>
    </div>
  )
}
