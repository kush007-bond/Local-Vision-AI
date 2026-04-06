# LocalVisionAI

> Local AI-powered video understanding pipeline — privacy-first, model-agnostic, no cloud required.

LocalVisionAI lets you point any local vision-capable AI model at a video file, live stream, webcam, or YouTube URL and receive timestamped natural language descriptions of what's happening — entirely on your machine.

---

## Features

- **Privacy first** — no frame, prompt, or result ever leaves your machine
- **Model agnostic** — Ollama, HuggingFace Transformers, llama.cpp, MLX (Apple Silicon)
- **Multiple input sources** — video files, webcam, RTSP streams, YouTube URLs, screen capture
- **Flexible frame sampling** — uniform FPS, keyframes-only, scene-change detection, adaptive
- **Multiple output formats** — console, JSON, SRT subtitles, CSV, SQLite timeline index
- **REST + WebSocket API** — built on FastAPI, stream results in real-time
- **CLI-first** — full-featured `localvisionai` command
- **Plugin system** — webhook callbacks, Discord notifications, and more

---

## Quick Start

### Prerequisites

- Python 3.10+
- [FFmpeg](https://www.ffmpeg.org/download.html) installed and added to PATH
- [Ollama](https://ollama.ai) installed (for the Ollama backend)

### Install

```bash
# Minimal install — Ollama backend only
pip install localvisionai[ollama]

# With HuggingFace Transformers support
pip install localvisionai[ollama,transformers]

# Full install (everything)
pip install localvisionai[all]

# Development install
git clone https://github.com/your-org/localvisionai
cd localvisionai
pip install -e ".[all,dev]"
```

### First Run

```bash
# Pull a vision model via Ollama
ollama pull gemma3

# Run on a video file
localvisionai run --video path/to/video.mp4 --model gemma3

# Run on your webcam
localvisionai run --source webcam --device 0 --model gemma3

# Run with scene-change-only sampling
localvisionai run --video path/to/video.mp4 --model gemma3 --sampler scene

# Export to JSON and SRT
localvisionai run --video path/to/video.mp4 --model gemma3 --output-formats json srt --output-dir ./results/

# Start the REST API server
localvisionai serve --host 127.0.0.1 --port 8765

# Search a processed video's timeline
localvisionai search --db results/timeline.db --query "person entering room"
```

---

## Supported Backends

| Backend | Flag | Models |
|---|---|---|
| **Ollama** | `--backend ollama` | gemma3, qwen2-vl, llava, llava-llama3, moondream |
| **HuggingFace** | `--backend transformers` | Qwen2-VL, LLaVA-1.5, InternVL2, MiniCPM-V, Gemma 3 |
| **llama.cpp** | `--backend llamacpp` | Any GGUF vision model |
| **MLX** | `--backend mlx` | Apple Silicon only — gemma3, llava, qwen2-vl |

---

## Hardware Requirements

| Setup | Minimum | Recommended |
|---|---|---|
| NVIDIA GPU | 6 GB VRAM (Q4 model) | RTX 3080 10 GB+ |
| Apple Silicon | M1 with 16 GB unified memory | M2 Pro+ |
| CPU only | 16 GB RAM | 32 GB RAM (slow, ~15s/frame) |

---

## Architecture

```
[Input Source] → [Frame Sampler] → [asyncio.Queue] → [Model Adapter] → [Output Handlers]
    │                 │                   │                  │                  │
  file/webcam/     uniform/scene/      bounded buffer     ollama/hf/        console/json/
  rtsp/url/screen  keyframe/adaptive   backpressure       llamacpp/mlx      srt/csv/sqlite
```

---

## Configuration

All settings can be expressed as CLI flags, environment variables (`LVA_` prefix), or a YAML config file:

```bash
localvisionai run --config configs/my_config.yaml
```

```yaml
# my_config.yaml
model:
  backend: ollama
  model_id: gemma3
sampling:
  strategy: scene
  fps: 1.0
output:
  formats: [console, json, srt]
  output_dir: ./results/
```

---

## Development

```bash
pip install -e ".[all,dev]"
pytest tests/unit/ -v
pytest tests/unit/ --cov=localvisionai --cov-report=term-missing
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add a new model adapter in 3 steps.
