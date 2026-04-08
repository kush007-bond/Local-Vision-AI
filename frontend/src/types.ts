export type Backend =
  | 'ollama'
  | 'openai'
  | 'anthropic'
  | 'gemini'
  | 'lmstudio'
  | 'transformers'
  | 'llamacpp'
  | 'mlx'

export type SourceType = 'file' | 'webcam' | 'rtsp' | 'url' | 'screen'
export type SamplerType = 'uniform' | 'scene' | 'keyframe' | 'adaptive'
export type JobStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'

export interface InferenceResult {
  timestamp: number
  description: string
  model_id: string
  backend: string
  latency_ms: number
  token_count: number
}

export interface Job {
  job_id: string
  status: JobStatus
  backend: string
  model_id: string
  source: string
  results: InferenceResult[]
  result_count: number
  error?: string
  started_at: string
  completed_at?: string
}

export interface JobSummary {
  job_id: string
  status: JobStatus
  backend: string
  model_id: string
  source: string
  result_count: number
  error?: string
  started_at: string
  completed_at?: string
}

export interface BackendInfo {
  id: Backend
  label: string
  type: 'local' | 'cloud'
  requires_api_key: boolean
  key_env_var?: string
  default_model: string
  example_models: string[]
  install: string
  note?: string
}

export interface JobConfig {
  backend: Backend
  model_id: string
  source_type: SourceType
  source_path?: string
  device_index?: number
  rtsp_url?: string
  fps: number
  sampler: SamplerType
  prompt: string
  system_prompt?: string
  context_mode: 'none' | 'sliding_window'
  api_key?: string
  api_base?: string
  max_tokens: number
  output_formats: string[]
  output_dir: string
}

export type WsMessage =
  | { type: 'result'; data: InferenceResult }
  | { type: 'status'; status: JobStatus }
  | { type: 'complete'; job_id: string; status: JobStatus }
  | { type: 'error'; message: string }
  | { type: 'ping' }

export type View = 'dashboard' | 'new-job' | 'job-detail'
