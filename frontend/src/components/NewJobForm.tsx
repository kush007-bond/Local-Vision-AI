import { useState, useEffect } from 'react'
import { Play, Eye, EyeOff, Info, AlertTriangle } from 'lucide-react'
import { api } from '../api'
import type { Backend, BackendInfo, JobConfig, SourceType, SamplerType } from '../types'
import { WebcamPreview } from './WebcamPreview'

const LOCAL_BACKENDS = new Set<Backend>(['ollama', 'lmstudio', 'transformers', 'llamacpp', 'mlx'])

// Samplers that are referenced in the UI but not yet implemented in the backend
const UNIMPLEMENTED_SAMPLERS = new Set<SamplerType>(['scene', 'adaptive'])

// Source types not yet implemented in the backend
const UNIMPLEMENTED_SOURCES = new Set<SourceType>(['rtsp', 'url', 'screen'])

interface Props {
  onJobCreated: (jobId: string, sourceType: SourceType, deviceIndex: number) => void
}

const DEFAULT: JobConfig = {
  backend: 'ollama',
  model_id: 'gemma3',
  source_type: 'file',
  fps: 1.0,
  sampler: 'uniform',
  prompt: 'Describe what is happening in this frame in one sentence.',
  context_mode: 'none',
  max_tokens: 512,
  output_formats: ['json'],
  output_dir: './output/',
  audio: false,
  audio_mode: 'transcribe',
  audio_window: 3.0,
  whisper_model: 'base',
}

function Field({ label, children, hint }: { label: string; children: React.ReactNode; hint?: string }) {
  return (
    <div>
      <label className="label">{label}</label>
      {children}
      {hint && <p className="mt-1 text-xs text-gray-600">{hint}</p>}
    </div>
  )
}

export function NewJobForm({ onJobCreated }: Props) {
  const [form, setForm]           = useState<JobConfig>(DEFAULT)
  const [backends, setBackends]   = useState<BackendInfo[]>([])
  const [showKey, setShowKey]     = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError]         = useState<string | null>(null)

  useEffect(() => {
    api.listBackends()
      .then(setBackends)
      .catch(() => {/* offline */})
  }, [])

  const set = <K extends keyof JobConfig>(key: K, val: JobConfig[K]) =>
    setForm(f => ({ ...f, [key]: val }))

  const currentBackend = backends.find(b => b.id === form.backend)
  const isCloud = !LOCAL_BACKENDS.has(form.backend)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError(null)
    setSubmitting(true)
    try {
      const job = await api.createJob(form)
      onJobCreated(job.job_id, form.source_type, form.device_index ?? 0)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <form onSubmit={e => void handleSubmit(e)} className="flex flex-col gap-6 p-6">
      <div>
        <h1 className="text-lg font-semibold text-gray-100">New Pipeline Job</h1>
        <p className="mt-0.5 text-sm text-gray-500">Configure a video source and AI model, then start the pipeline.</p>
      </div>

      <div className="grid grid-cols-2 gap-6">
        {/* ── Left column: Source ──────────────────────────── */}
        <div className="flex flex-col gap-4">
          <h2 className="text-xs font-semibold uppercase tracking-widest text-gray-500">Input Source</h2>

          <Field label="Source type">
            <select
              className="select"
              value={form.source_type}
              onChange={e => {
                const t = e.target.value as SourceType
                setForm(f => ({
                  ...f,
                  source_type: t,
                  // Auto-enable audio flag and switch prompt for audio-only mode
                  audio: t === 'audio' ? true : f.audio,
                  prompt: t === 'audio'
                    ? 'Summarise what is being said in the audio transcript in one sentence.'
                    : (f.source_type === 'audio'
                        ? 'Describe what is happening in this frame in one sentence.'
                        : f.prompt),
                }))
              }}
            >
              <option value="file">Video file</option>
              <option value="webcam">Webcam</option>
              <option value="rtsp">RTSP stream</option>
              <option value="url">YouTube / URL</option>
              <option value="screen">Screen capture</option>
              <option value="audio">🎵 Audio only</option>
            </select>
          </Field>

          {form.source_type === 'file' && (
            <Field label="File path" hint="Absolute path or path relative to the server working directory">
              <input
                className="input"
                placeholder="/path/to/video.mp4"
                value={form.source_path ?? ''}
                onChange={e => set('source_path', e.target.value)}
                required
              />
            </Field>
          )}

          {form.source_type === 'webcam' && (
            <>
              <Field label="Device index" hint="0 = default webcam">
                <input
                  className="input"
                  type="number"
                  min={0}
                  value={form.device_index ?? 0}
                  onChange={e => set('device_index', Number(e.target.value))}
                />
              </Field>
              <WebcamPreview deviceIndex={form.device_index ?? 0} />
            </>
          )}

          {form.source_type === 'rtsp' && (
            <Field label="RTSP URL">
              <input
                className="input"
                placeholder="rtsp://user:pass@192.168.1.1/stream"
                value={form.rtsp_url ?? ''}
                onChange={e => set('rtsp_url', e.target.value)}
                required
              />
            </Field>
          )}

          {form.source_type === 'url' && (
            <Field label="Video URL" hint="YouTube, direct MP4, or any yt-dlp supported URL">
              <input
                className="input"
                placeholder="https://youtube.com/watch?v=..."
                value={form.source_path ?? ''}
                onChange={e => set('source_path', e.target.value)}
                required
              />
            </Field>
          )}

          {form.source_type === 'audio' && (
            <>
              <Field label="Audio / video file path" hint="Path to an MP3, WAV, MP4, or any FFmpeg-supported file">
                <input
                  className="input"
                  placeholder="/path/to/audio.mp3"
                  value={form.source_path ?? ''}
                  onChange={e => set('source_path', e.target.value)}
                  required
                />
              </Field>
              <Field label="Whisper model size" hint="Larger = more accurate but slower">
                <select
                  className="select"
                  value={form.whisper_model ?? 'base'}
                  onChange={e => set('whisper_model', e.target.value as JobConfig['whisper_model'])}
                >
                  <option value="tiny">tiny (fastest)</option>
                  <option value="base">base (recommended)</option>
                  <option value="small">small</option>
                  <option value="medium">medium</option>
                  <option value="large">large (slowest, most accurate)</option>
                </select>
              </Field>
            </>
          )}

          {UNIMPLEMENTED_SOURCES.has(form.source_type) && (
            <div className="flex items-start gap-2 rounded-lg border border-amber-800 bg-amber-950/40 px-3 py-2 text-xs text-amber-400">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <span>This source type is not yet implemented and the job will fail at runtime.</span>
            </div>
          )}

          <h2 className="pt-2 text-xs font-semibold uppercase tracking-widest text-gray-500">Sampling</h2>

          <div className="grid grid-cols-2 gap-3">
            <Field label="Strategy">
              <select
                className="select"
                value={form.sampler}
                onChange={e => set('sampler', e.target.value as SamplerType)}
              >
                <option value="uniform">Uniform FPS</option>
                <option value="keyframe">Keyframes only</option>
                <option value="scene">Scene changes (not implemented)</option>
                <option value="adaptive">Adaptive (not implemented)</option>
              </select>
              {UNIMPLEMENTED_SAMPLERS.has(form.sampler) && (
                <div className="mt-1 flex items-center gap-1.5 text-xs text-amber-400">
                  <AlertTriangle className="h-3 w-3 shrink-0" />
                  This sampler is not yet implemented and the job will fail at runtime.
                </div>
              )}
            </Field>

            <Field label={`FPS (${form.fps})`}>
              <input
                className="input"
                type="range"
                min={0.1}
                max={5}
                step={0.1}
                value={form.fps}
                onChange={e => set('fps', Number(e.target.value))}
                style={{ paddingLeft: 0, paddingRight: 0, cursor: 'pointer' }}
              />
            </Field>
          </div>

          <Field label="Context mode" hint="Sliding window feeds prior descriptions to the model">
            <select
              className="select"
              value={form.context_mode}
              onChange={e => set('context_mode', e.target.value as 'none' | 'sliding_window')}
            >
              <option value="none">None</option>
              <option value="sliding_window">Sliding window</option>
            </select>
          </Field>
        </div>

        {/* ── Right column: Model ──────────────────────────── */}
        <div className="flex flex-col gap-4">
          <h2 className="text-xs font-semibold uppercase tracking-widest text-gray-500">Model</h2>

          <Field label="Backend">
            <select
              className="select"
              value={form.backend}
              onChange={e => {
                const b = e.target.value as Backend
                const info = backends.find(x => x.id === b)
                set('backend', b)
                if (info) set('model_id', info.default_model)
              }}
            >
              {backends.length > 0
                ? backends.map(b => (
                    <option key={b.id} value={b.id}>
                      {b.label} {b.type === 'local' ? '(local)' : '(cloud)'}
                    </option>
                  ))
                : (
                  <>
                    <option value="ollama">Ollama (local)</option>
                    <option value="openai">OpenAI (cloud)</option>
                    <option value="anthropic">Anthropic (cloud)</option>
                    <option value="gemini">Google Gemini (cloud)</option>
                    <option value="lmstudio">LM Studio (local)</option>
                    <option value="transformers">HuggingFace (local)</option>
                  </>
                )
              }
            </select>
          </Field>

          {currentBackend && (
            <div className="flex items-start gap-2 rounded-lg border border-bg-border bg-bg-elevated p-3 text-xs text-gray-500">
              <Info className="mt-0.5 h-3.5 w-3.5 shrink-0 text-gray-600" />
              <div className="space-y-0.5">
                {currentBackend.note && <p>{currentBackend.note}</p>}
                <p>Install: <code className="font-mono text-gray-400">{currentBackend.install}</code></p>
              </div>
            </div>
          )}

          <Field
            label="Model name"
            hint={currentBackend?.example_models.slice(0, 3).join(' · ')}
          >
            <input
              className="input"
              value={form.model_id}
              onChange={e => set('model_id', e.target.value)}
              required
            />
          </Field>

          {isCloud && (
            <Field
              label="API Key"
              hint={currentBackend?.key_env_var ? `Or set ${currentBackend.key_env_var} env var` : undefined}
            >
              <div className="relative">
                <input
                  className="input pr-10"
                  type={showKey ? 'text' : 'password'}
                  placeholder="sk-..."
                  value={form.api_key ?? ''}
                  onChange={e => set('api_key', e.target.value || undefined)}
                />
                <button
                  type="button"
                  onClick={() => setShowKey(v => !v)}
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300"
                >
                  {showKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
            </Field>
          )}

          <Field
            label="API base URL"
            hint={form.backend === 'lmstudio' ? 'Default: http://localhost:1234/v1' : 'Leave blank to use the default endpoint'}
          >
            <input
              className="input"
              placeholder={form.backend === 'ollama' ? 'http://localhost:11434' : form.backend === 'lmstudio' ? 'http://localhost:1234/v1' : 'https://api.openai.com/v1'}
              value={form.api_base ?? ''}
              onChange={e => set('api_base', e.target.value || undefined)}
            />
          </Field>

          {isCloud && (
            <Field label={`Max tokens (${form.max_tokens})`}>
              <input
                className="input"
                type="range"
                min={64}
                max={2048}
                step={64}
                value={form.max_tokens}
                onChange={e => set('max_tokens', Number(e.target.value))}
                style={{ paddingLeft: 0, paddingRight: 0, cursor: 'pointer' }}
              />
            </Field>
          )}

          <h2 className="pt-2 text-xs font-semibold uppercase tracking-widest text-gray-500">Prompt</h2>

          <Field label="User prompt">
            <textarea
              className="input min-h-[80px] resize-y"
              value={form.prompt}
              onChange={e => set('prompt', e.target.value)}
              required
            />
          </Field>

          <Field label="Output directory">
            <input
              className="input"
              value={form.output_dir}
              onChange={e => set('output_dir', e.target.value)}
            />
          </Field>
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-red-800 bg-red-950/50 px-4 py-3 text-sm text-red-400">
          {error}
        </div>
      )}

      <div className="flex justify-end">
        <button type="submit" disabled={submitting} className="btn-primary px-6">
          {submitting
            ? <><span className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" /> Starting…</>
            : <><Play className="h-4 w-4" /> Start Pipeline</>
          }
        </button>
      </div>
    </form>
  )
}
