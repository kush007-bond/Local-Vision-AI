import { LayoutDashboard, Plus, type LucideIcon } from 'lucide-react'
import type { View } from '../types'

interface NavItem {
  id: View
  label: string
  icon: LucideIcon
}

const NAV: NavItem[] = [
  { id: 'dashboard', label: 'Dashboard',  icon: LayoutDashboard },
  { id: 'new-job',   label: 'New Job',    icon: Plus },
]

interface Props {
  current: View
  onChange: (v: View) => void
  activeCount: number
}

export function Sidebar({ current, onChange, activeCount }: Props) {
  return (
    <aside className="flex w-52 shrink-0 flex-col border-r border-bg-border bg-bg-surface py-4">
      <nav className="flex flex-col gap-1 px-3">
        {NAV.map(({ id, label, icon: Icon }) => {
          const active = current === id || (current === 'job-detail' && id === 'dashboard')
          return (
            <button
              key={id}
              onClick={() => onChange(id)}
              className={`flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                active
                  ? 'bg-brand/15 text-brand'
                  : 'text-gray-400 hover:bg-bg-elevated hover:text-gray-100'
              }`}
            >
              <Icon className="h-4 w-4 shrink-0" />
              {label}
              {id === 'dashboard' && activeCount > 0 && (
                <span className="ml-auto rounded-full bg-brand px-1.5 py-0.5 text-xs font-semibold text-white leading-none">
                  {activeCount}
                </span>
              )}
            </button>
          )
        })}
      </nav>

      {/* Bottom info */}
      <div className="mt-auto px-4">
        <p className="text-xs text-gray-600">v0.1.0 · Privacy-first</p>
      </div>
    </aside>
  )
}
