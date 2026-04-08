import { Eye, Activity } from 'lucide-react'

interface Props {
  serverOnline: boolean
}

export function Header({ serverOnline }: Props) {
  return (
    <header className="flex h-14 shrink-0 items-center gap-3 border-b border-bg-border bg-bg-surface px-5">
      {/* Logo */}
      <div className="flex items-center gap-2.5">
        <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-brand/20">
          <Eye className="h-4 w-4 text-brand" />
        </div>
        <span className="text-sm font-semibold tracking-tight text-gray-100">
          LocalVision<span className="text-brand">AI</span>
        </span>
      </div>

      <div className="mx-3 h-5 w-px bg-bg-border" />

      <span className="text-xs text-gray-500">Local video understanding pipeline</span>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Server status */}
      <div className="flex items-center gap-2 rounded-full border border-bg-border bg-bg-elevated px-3 py-1">
        <Activity className="h-3.5 w-3.5 text-gray-500" />
        <span className={`text-xs font-medium ${serverOnline ? 'text-emerald-400' : 'text-red-400'}`}>
          {serverOnline ? 'Server online' : 'Server offline'}
        </span>
        <span
          className={`h-1.5 w-1.5 rounded-full ${
            serverOnline ? 'bg-emerald-400 shadow-[0_0_4px] shadow-emerald-400' : 'bg-red-500'
          }`}
        />
      </div>
    </header>
  )
}
