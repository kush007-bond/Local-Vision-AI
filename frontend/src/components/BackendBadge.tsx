import type { Backend } from '../types'

const COLORS: Record<string, string> = {
  ollama:       'bg-purple-950 text-purple-300',
  openai:       'bg-emerald-950 text-emerald-300',
  anthropic:    'bg-orange-950 text-orange-300',
  gemini:       'bg-blue-950 text-blue-300',
  lmstudio:     'bg-violet-950 text-violet-300',
  transformers: 'bg-yellow-950 text-yellow-300',
  llamacpp:     'bg-rose-950 text-rose-300',
  mlx:          'bg-sky-950 text-sky-300',
}

export function BackendBadge({ backend }: { backend: string }) {
  const color = COLORS[backend] ?? 'bg-zinc-800 text-zinc-300'
  return (
    <span className={`rounded-md px-2 py-0.5 font-mono text-xs font-medium ${color}`}>
      {backend}
    </span>
  )
}
