# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What's implemented vs. still missing

**Fully implemented:**
- Core pipeline, config, CLI, prompt/context system
- `OllamaAdapter`, `HuggingFaceAdapter`, `OpenAIAdapter`, `AnthropicAdapter`, `GeminiAdapter`, `LMStudioAdapter`
- `VideoFileSource`, `UniformSampler`, `KeyframeSampler`
- `ConsoleOutput`, `JSONOutput`, `WebSocketOutput`
- FastAPI REST + WebSocket server (`localvisionai/api/server.py`)
- React + TypeScript + Tailwind + Vite frontend (`frontend/`)
- `AnalysisPipeline` (`localvisionai/analysis/`) — offline 4-phase workflow: frame analysis → full audio transcription (Whisper) → summary → interactive Q&A. Cache stored at `<output_dir>/<stem>_analysis.json`.
- Webcam live preview in `NewJobForm.tsx` — uses browser `getUserMedia`, renders in a 16:9 `<video>` element, starts/stops independently of the pipeline job.

**Referenced but not yet implemented (will raise ImportError at runtime):**
- Input sources: `RTSPSource`, `URLSource`, `ScreenCaptureSource`
- Samplers: `SceneSampler`, `AdaptiveSampler`
- Output handlers: `SRTOutput`, `CSVOutput`, `SQLiteOutput`
- Local adapters: `LlamaCppAdapter`, `MLXAdapter`, `VLLMAdapter`

## Development Setup

```bash
# Development install (minimal — Ollama + dev tools)
pip install -e ".[ollama,dev]"

# Full install (all backends and optional features)
pip install -e ".[all,dev]"
```

## Frontend Setup

```bash
cd frontend
npm install
npm run dev        # dev server at http://localhost:5173 (proxies to API at :8765)
npm run build      # builds to frontend/dist/ — served automatically by localvisionai serve
```

## Commands

```bash
# Run all unit tests
pytest tests/unit/ -v

# Run unit tests with coverage
pytest tests/unit/ --cov=localvisionai --cov-report=term-missing

# Run a single test file
pytest tests/unit/test_pipeline.py -v

# Run a single test by name
pytest tests/unit/test_pipeline.py::TestPipeline::test_pipeline_calls_handler_for_each_frame -v

# Skip integration tests (requires real hardware/models)
pytest tests/unit/ -m "not integration"

# Lint
ruff check localvisionai/ tests/
```

## Architecture

The pipeline follows an async producer-consumer pattern with a bounded `asyncio.Queue` providing backpressure:

```
[Input Source] → [Frame Sampler] → [asyncio.Queue] → [Model Adapter] → [Output Handlers]
```

**Core flow** (`localvisionai/pipeline.py`):
- `Pipeline.run()` builds all components from `PipelineConfig`, opens handlers, loads the adapter, then runs `_producer` and `_consumer` concurrently via `asyncio.gather`.
- `_producer`: streams frames from the source, applies the sampler's `should_process()` gate, and enqueues `(frame, timestamp)` tuples.
- `_consumer`: dequeues frames, calls `build_prompt()` (with optional sliding-window context), streams tokens from the adapter, assembles an `InferenceResult`, and fans out to all output handlers.

**Key modules:**
- `localvisionai/config.py` — Pydantic v2 config hierarchy (`PipelineConfig` → sub-configs). Config priority: CLI flags > `LVA_*` env vars > YAML > defaults.
- `localvisionai/adapters/` — `AbstractModelAdapter` defines the interface (`load`, `unload`, `infer`, `infer_multi`). `registry.py` maps backend name strings to adapter classes; `llamacpp`, `mlx`, and `vllm` are lazy-imported to avoid requiring optional packages.
- `localvisionai/inputs/` — Input sources (`file`, `webcam`, `rtsp`, `url`, `screen`) all expose an async context manager with a `stream()` async generator yielding `(PIL.Image, float)`.
- `localvisionai/sampling/` — Samplers implement `should_process(frame, timestamp) -> bool`. Built via `build_sampler(SamplingConfig)`.
- `localvisionai/outputs/` — Output handlers implement `open(job_id)`, `handle(InferenceResult)`, `close()`. Built via `build_handlers(OutputConfig, job_id)`.
- `localvisionai/prompts/` — `build_prompt()` constructs the (user_prompt, system_prompt) tuple. `ContextWindow` maintains a token-budget sliding window of prior descriptions for the `sliding_window` context mode.

## Supported Backends

| Backend | Flag | Install extra | Auth |
|---|---|---|---|
| Ollama (local) | `--backend ollama` | `[ollama]` | none — `ollama serve` must be running |
| HuggingFace (local) | `--backend transformers` | `[transformers]` | none |
| OpenAI | `--backend openai` | `[openai]` | `OPENAI_API_KEY` env var or `--api-key` |
| Anthropic/Claude | `--backend anthropic` | `[anthropic]` | `ANTHROPIC_API_KEY` env var or `--api-key` |
| Google Gemini | `--backend gemini` | `[gemini]` | `GOOGLE_API_KEY` env var or `--api-key` |
| LM Studio (local) | `--backend lmstudio` | `[openai]` | none — LM Studio server must be running |
| llama.cpp | `--backend llamacpp` | `[llamacpp]` | none |
| MLX (Apple Silicon) | `--backend mlx` | `[mlx]` | none |

Cloud backend CLI examples:
```bash
# OpenAI
localvisionai run --video v.mp4 --backend openai --model gpt-4o

# Anthropic Claude
localvisionai run --video v.mp4 --backend anthropic --model claude-sonnet-4-6

# Google Gemini
localvisionai run --video v.mp4 --backend gemini --model gemini-2.0-flash

# LM Studio (model name must match what's loaded in LM Studio)
localvisionai run --video v.mp4 --backend lmstudio --model llava-v1.6-mistral-7b
# Custom endpoint:
localvisionai run --video v.mp4 --backend lmstudio --model mymodel --api-base http://192.168.1.5:1234/v1
```

## Adding a New Backend Adapter

1. Create `localvisionai/adapters/<name>_adapter.py` implementing `AbstractModelAdapter`.
2. Add it to `_LAZY_ADAPTERS` in `localvisionai/adapters/registry.py`.
3. Add the optional dependency under `[project.optional-dependencies]` in `pyproject.toml`.

## Known Issues

- `frontend/src/components/BackendBadge.tsx` has a pre-existing `TS6133` unused-import error that causes `npm run build` to fail — not a regression, fix before shipping frontend.

## Testing Conventions

- Unit tests mock the adapter, source, and handlers via `unittest.mock`. See `tests/unit/test_pipeline.py` for the patch pattern (`build_source`, `get_adapter`, `build_handlers`).
- Integration tests are marked `@pytest.mark.integration` and excluded from CI by default.
- `asyncio_mode = "auto"` is set in `pyproject.toml`, so async test functions don't need `@pytest.mark.asyncio` explicitly (though it's used for clarity).
