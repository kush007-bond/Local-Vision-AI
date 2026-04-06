# Technical Requirements Document (TRD)
## LocalVisionAI — Local AI-Powered Video Understanding Pipeline

**Version:** 1.0.0
**Status:** Draft
**Last Updated:** 2026-04-07
**Owner:** Open Source Community
**Document Type:** Technical Requirements Document

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Technology Stack](#2-technology-stack)
3. [Project Structure](#3-project-structure)
4. [Architecture & Data Flow](#4-architecture--data-flow)
5. [Core Components](#5-core-components)
6. [Model Adapter Interface](#6-model-adapter-interface)
7. [API Design](#7-api-design)
8. [CLI Design](#8-cli-design)
9. [Configuration System](#9-configuration-system)
10. [Storage & Persistence](#10-storage--persistence)
11. [Async Pipeline Design](#11-async-pipeline-design)
12. [Hardware Abstraction](#12-hardware-abstraction)
13. [Plugin System](#13-plugin-system)
14. [Error Handling & Resilience](#14-error-handling--resilience)
15. [Testing Strategy](#15-testing-strategy)
16. [Performance Targets & Benchmarks](#16-performance-targets--benchmarks)
17. [Security Considerations](#17-security-considerations)
18. [Dependency Matrix](#18-dependency-matrix)
19. [Environment & Installation](#19-environment--installation)
20. [Contribution Guide for Adapters](#20-contribution-guide-for-adapters)

---

## 1. System Overview

LocalVisionAI is a Python-only, async-native pipeline with four logical layers:

```
[Input Layer]     → Video files, streams, URLs, screen capture
[Processing Layer] → Frame decoding, sampling, resizing, queuing
[Inference Layer] → Model-agnostic vision AI runner
[Output Layer]    → Structured results, API, CLI, storage
```

The pipeline is fully asynchronous using `asyncio`. A producer (frame extractor) and consumer (model runner) communicate via a bounded `asyncio.Queue`. This prevents memory overflow and implements natural backpressure — the extractor slows or drops frames if the model is slower than the source video.

All components are Python classes with abstract base classes (ABCs) defining the interface contract. Swapping any component — the input source, the frame sampler, the model backend, or the output handler — requires only a configuration change, not a code change.

---

## 2. Technology Stack

### Language & Runtime

| Concern | Choice | Reason |
|---|---|---|
| Language | Python 3.10+ | Universal ML ecosystem support |
| Async runtime | `asyncio` (stdlib) | Zero-dependency async I/O |
| Type system | `typing` + `dataclasses` | Static analysis, IDE support |
| Packaging | `pyproject.toml` + `hatch` | Modern Python packaging standard |

### Video Decoding

| Concern | Choice | Fallback |
|---|---|---|
| Primary decoder | `av` (PyAV, FFmpeg bindings) | `opencv-python` |
| Scene detection | `scenedetect` | Manual frame diff via `numpy` |
| Frame format | `PIL.Image` (Pillow) | `numpy` array |
| Remote URLs | `yt-dlp` (subprocess) | `urllib` for direct HTTP |
| Screen capture | `mss` | `PIL.ImageGrab` |

### Vision Model Backends

| Backend | Library | Hardware |
|---|---|---|
| Ollama | `ollama` (official Python SDK) | CUDA, Metal, CPU |
| llama.cpp | `llama-cpp-python` | CUDA, Metal, CPU |
| HuggingFace | `transformers`, `accelerate`, `bitsandbytes` | CUDA, CPU |
| MLX | `mlx`, `mlx-vlm` | Apple Silicon only |
| vLLM | `vllm` | CUDA only |

### API & Interfaces

| Concern | Choice |
|---|---|
| REST + WebSocket API | `fastapi` + `uvicorn` |
| CLI | `typer` |
| Web UI | `gradio` |
| Serialization | `pydantic` v2 |

### Storage

| Concern | Choice |
|---|---|
| Timeline index | `sqlite3` (stdlib) + `aiosqlite` |
| Embedding store (RAG) | `chromadb` (optional) |
| Config files | `pyyaml` |
| Export formats | `json` (stdlib), `csv` (stdlib) |

### OCR & Audio (optional modules)

| Concern | Choice |
|---|---|
| OCR | `pytesseract` + Tesseract binary |
| Speech transcription | `openai-whisper` (local, no cloud) |
| Similarity / embeddings | `sentence-transformers` |

---

## 3. Project Structure

```
localvisionai/
├── pyproject.toml
├── README.md
├── CONTRIBUTING.md
├── LICENSE
├── docs/
│   ├── getting-started.md
│   ├── model-adapters.md
│   ├── api-reference.md
│   └── plugins.md
├── configs/
│   └── default.yaml
├── tests/
│   ├── unit/
│   │   ├── test_sampler.py
│   │   ├── test_pipeline.py
│   │   ├── test_adapters.py
│   │   └── test_outputs.py
│   ├── integration/
│   │   ├── test_ollama_backend.py
│   │   └── test_api_endpoints.py
│   └── fixtures/
│       ├── sample_video.mp4          # 10-second test video
│       └── mock_model_response.json
├── localvisionai/
│   ├── __init__.py
│   ├── cli.py                        # Typer CLI entrypoint
│   ├── pipeline.py                   # Orchestrates all components
│   ├── config.py                     # Config loading and validation (Pydantic)
│   │
│   ├── inputs/
│   │   ├── __init__.py
│   │   ├── base.py                   # AbstractVideoSource
│   │   ├── file_source.py            # VideoFileSource
│   │   ├── webcam_source.py          # WebcamSource
│   │   ├── rtsp_source.py            # RTSPSource
│   │   ├── url_source.py             # URLSource (yt-dlp)
│   │   └── screen_source.py          # ScreenCaptureSource
│   │
│   ├── sampling/
│   │   ├── __init__.py
│   │   ├── base.py                   # AbstractFrameSampler
│   │   ├── uniform_sampler.py        # Fixed FPS extraction
│   │   ├── keyframe_sampler.py       # I-frames only
│   │   ├── scene_sampler.py          # Scene change detection
│   │   └── adaptive_sampler.py       # Motion-adaptive FPS
│   │
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── base.py                   # AbstractModelAdapter
│   │   ├── ollama_adapter.py
│   │   ├── llamacpp_adapter.py
│   │   ├── transformers_adapter.py
│   │   ├── mlx_adapter.py
│   │   ├── vllm_adapter.py
│   │   └── registry.py               # Auto-discovers adapters
│   │
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── templates.py              # Per-model chat templates
│   │   ├── builder.py                # Prompt construction logic
│   │   └── memory.py                 # Sliding window context
│   │
│   ├── outputs/
│   │   ├── __init__.py
│   │   ├── base.py                   # AbstractOutputHandler
│   │   ├── console_output.py
│   │   ├── json_output.py
│   │   ├── srt_output.py
│   │   ├── csv_output.py
│   │   └── sqlite_output.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── server.py                 # FastAPI app
│   │   ├── routes/
│   │   │   ├── jobs.py               # Job submission, status, results
│   │   │   ├── stream.py             # WebSocket streaming endpoint
│   │   │   └── models.py             # Installed model listing
│   │   └── schemas.py                # Pydantic request/response models
│   │
│   ├── search/
│   │   ├── __init__.py
│   │   ├── indexer.py                # SQLite full-text indexer
│   │   └── rag.py                    # ChromaDB-based semantic search
│   │
│   ├── plugins/
│   │   ├── __init__.py
│   │   ├── base.py                   # AbstractPlugin
│   │   ├── loader.py                 # Dynamic plugin discovery
│   │   └── builtin/
│   │       ├── webhook_plugin.py
│   │       └── discord_plugin.py
│   │
│   ├── extras/
│   │   ├── ocr.py                    # Tesseract OCR integration
│   │   ├── whisper.py                # Whisper audio transcription
│   │   └── tracker.py               # Cross-frame entity tracking
│   │
│   └── utils/
│       ├── logging.py                # Structured logging setup
│       ├── hardware.py               # GPU/CPU detection
│       ├── image.py                  # Frame resize / encode helpers
│       └── timing.py                 # Latency profiling utilities
```

---

## 4. Architecture & Data Flow

### 4.1 Full Data Flow

```
Video Source
    │
    ▼
AbstractVideoSource.stream()  ──► yields raw frames as PIL.Image
    │
    ▼
AbstractFrameSampler.should_process(frame, timestamp)
    │  filters by FPS / scene change / keyframe
    ▼
Frame → resize to model resolution → encode to base64 or bytes
    │
    ▼
asyncio.Queue(maxsize=N)       ──► bounded buffer, backpressure
    │
    ▼
AbstractModelAdapter.infer(frame, prompt, context)
    │  returns AsyncGenerator[str, None] (token stream)
    ▼
FrameResult(timestamp, description, model, latency_ms, raw_tokens)
    │
    ▼
AbstractOutputHandler.handle(result)
    │  fan-out to all registered handlers
    ▼
[ConsoleOutput] [JSONOutput] [SRTOutput] [SQLiteOutput] [PluginCallbacks]
```

### 4.2 Async Producer-Consumer Pattern

```python
# Conceptual model — actual implementation in pipeline.py

async def run(config: PipelineConfig):
    queue = asyncio.Queue(maxsize=config.queue_size)

    source = build_source(config)
    sampler = build_sampler(config)
    adapter = build_adapter(config)
    handlers = build_handlers(config)

    async def producer():
        async for frame, ts in source.stream():
            if sampler.should_process(frame, ts):
                await queue.put((frame, ts))
        await queue.put(None)  # sentinel

    async def consumer():
        context = ContextWindow(max_tokens=config.context_tokens)
        while True:
            item = await queue.get()
            if item is None:
                break
            frame, ts = item
            prompt = build_prompt(config, context)
            result = await adapter.infer(frame, prompt)
            context.update(result)
            for handler in handlers:
                await handler.handle(result)

    await asyncio.gather(producer(), consumer())
```

### 4.3 Multi-frame Context Window

When `context_mode = "sliding_window"`, the prompt builder injects a rolling summary of the last N frame descriptions into each new prompt:

```
System: You are describing video frames in sequence.
Previous context: A person walked into the room carrying a bag. They placed
the bag on a desk and opened a laptop.
Current task: Describe what is happening in this new frame.
```

This gives single-image models temporal awareness without requiring native video input support.

---

## 5. Core Components

### 5.1 AbstractVideoSource

```python
# localvisionai/inputs/base.py

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Tuple
from PIL import Image

class AbstractVideoSource(ABC):
    """Base class for all video input sources."""

    @abstractmethod
    async def stream(self) -> AsyncGenerator[Tuple[Image.Image, float], None]:
        """
        Yields (frame, timestamp_seconds) tuples.
        Frame is a PIL.Image. Timestamp is seconds from start.
        Must be an async generator.
        """
        ...

    @abstractmethod
    async def open(self) -> None:
        """Initialize any resources (file handles, connections)."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release all resources cleanly."""
        ...

    @property
    @abstractmethod
    def metadata(self) -> dict:
        """Returns {duration, fps, width, height, codec} where available."""
        ...
```

### 5.2 VideoFileSource (PyAV implementation)

```python
# localvisionai/inputs/file_source.py

import av
import asyncio
from PIL import Image
from .base import AbstractVideoSource

class VideoFileSource(AbstractVideoSource):
    def __init__(self, path: str):
        self.path = path
        self._container = None

    async def open(self):
        # av.open is synchronous; run in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        self._container = await loop.run_in_executor(None, av.open, self.path)

    async def close(self):
        if self._container:
            self._container.close()

    @property
    def metadata(self) -> dict:
        stream = self._container.streams.video[0]
        return {
            "duration": float(self._container.duration / 1_000_000),
            "fps": float(stream.average_rate),
            "width": stream.width,
            "height": stream.height,
            "codec": stream.codec_context.name,
        }

    async def stream(self):
        loop = asyncio.get_event_loop()
        stream = self._container.streams.video[0]

        def _decode_frames():
            for packet in self._container.demux(stream):
                for frame in packet.decode():
                    ts = float(frame.pts * stream.time_base)
                    img = frame.to_image()
                    yield img, ts

        gen = _decode_frames()
        while True:
            try:
                frame, ts = await loop.run_in_executor(None, next, gen)
                yield frame, ts
            except StopIteration:
                break
```

### 5.3 AbstractFrameSampler

```python
# localvisionai/sampling/base.py

from abc import ABC, abstractmethod
from PIL import Image

class AbstractFrameSampler(ABC):
    @abstractmethod
    def should_process(self, frame: Image.Image, timestamp: float) -> bool:
        """Return True if this frame should be sent for inference."""
        ...

    def reset(self):
        """Reset internal state (e.g. for new video)."""
        pass
```

### 5.4 UniformSampler

```python
# localvisionai/sampling/uniform_sampler.py

class UniformSampler(AbstractFrameSampler):
    def __init__(self, fps: float = 1.0):
        self.interval = 1.0 / fps
        self._last_sent = -self.interval

    def should_process(self, frame, timestamp: float) -> bool:
        if timestamp - self._last_sent >= self.interval:
            self._last_sent = timestamp
            return True
        return False
```

### 5.5 SceneSampler

```python
# localvisionai/sampling/scene_sampler.py

import numpy as np
from PIL import Image

class SceneSampler(AbstractFrameSampler):
    def __init__(self, threshold: float = 30.0, min_interval: float = 0.5):
        self.threshold = threshold          # Mean absolute difference threshold
        self.min_interval = min_interval   # Minimum seconds between sends
        self._last_frame_array = None
        self._last_sent = -min_interval

    def should_process(self, frame: Image.Image, timestamp: float) -> bool:
        if timestamp - self._last_sent < self.min_interval:
            return False

        arr = np.array(frame.convert("L").resize((64, 64)))  # Small grayscale

        if self._last_frame_array is None:
            self._last_frame_array = arr
            self._last_sent = timestamp
            return True

        diff = np.mean(np.abs(arr.astype(float) - self._last_frame_array.astype(float)))
        self._last_frame_array = arr

        if diff > self.threshold:
            self._last_sent = timestamp
            return True
        return False
```

---

## 6. Model Adapter Interface

### 6.1 AbstractModelAdapter

Every model backend implements this interface. This is the single most important contract in the codebase.

```python
# localvisionai/adapters/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncGenerator
from PIL import Image

@dataclass
class InferenceResult:
    timestamp: float
    description: str          # Full assembled description
    model_id: str
    backend: str
    latency_ms: float
    token_count: int
    raw_tokens: list[str]     # Individual streamed tokens

class AbstractModelAdapter(ABC):

    @abstractmethod
    async def load(self) -> None:
        """Load model into memory. Called once at pipeline start."""
        ...

    @abstractmethod
    async def unload(self) -> None:
        """Release model from memory. Called on pipeline shutdown."""
        ...

    @abstractmethod
    async def infer(
        self,
        frame: Image.Image,
        prompt: str,
        system_prompt: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Async generator that yields tokens as they stream from the model.
        Caller collects tokens into a full description.
        """
        ...

    @abstractmethod
    async def infer_multi(
        self,
        frames: list[Image.Image],
        prompt: str,
        system_prompt: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Multi-frame inference for models that support it (Qwen2-VL, etc).
        Falls back to single-frame if not supported.
        """
        ...

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Return canonical model identifier."""
        ...

    @property
    def supports_multi_frame(self) -> bool:
        """Override to True in adapters that natively support multiple images."""
        return False

    @property
    def preferred_resolution(self) -> tuple[int, int]:
        """Return (width, height) that this model performs best at."""
        return (448, 448)
```

### 6.2 OllamaAdapter

```python
# localvisionai/adapters/ollama_adapter.py

import ollama
import io
import time
import base64
from PIL import Image
from .base import AbstractModelAdapter

class OllamaAdapter(AbstractModelAdapter):
    def __init__(self, model: str, host: str = "http://localhost:11434"):
        self._model = model
        self._client = ollama.AsyncClient(host=host)

    @property
    def model_id(self) -> str:
        return self._model

    async def load(self):
        # Ollama manages model loading; just verify it's available
        models = await self._client.list()
        names = [m["name"] for m in models.get("models", [])]
        if self._model not in names and f"{self._model}:latest" not in names:
            raise RuntimeError(
                f"Model '{self._model}' not found in Ollama. "
                f"Run: ollama pull {self._model}"
            )

    async def unload(self):
        pass  # Ollama manages its own lifecycle

    def _encode_frame(self, frame: Image.Image) -> str:
        buf = io.BytesIO()
        frame.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()

    async def infer(self, frame, prompt, system_prompt=None):
        img_b64 = self._encode_frame(frame)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": prompt,
            "images": [img_b64]
        })

        async for chunk in await self._client.chat(
            model=self._model,
            messages=messages,
            stream=True,
        ):
            token = chunk["message"]["content"]
            if token:
                yield token

    async def infer_multi(self, frames, prompt, system_prompt=None):
        encoded = [self._encode_frame(f) for f in frames]
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": prompt,
            "images": encoded
        })
        async for chunk in await self._client.chat(
            model=self._model,
            messages=messages,
            stream=True,
        ):
            token = chunk["message"]["content"]
            if token:
                yield token
```

### 6.3 HuggingFaceAdapter

```python
# localvisionai/adapters/transformers_adapter.py

import asyncio
from PIL import Image
from .base import AbstractModelAdapter

class HuggingFaceAdapter(AbstractModelAdapter):
    """
    Supports any HuggingFace model with a visual instruction-following interface.
    Tested with: Qwen2-VL, LLaVA-1.5, InternVL2, MiniCPM-V, Gemma3.
    """

    def __init__(self, model_id: str, device: str = "auto", load_in_4bit: bool = False):
        self._model_id = model_id
        self._device = device
        self._load_in_4bit = load_in_4bit
        self._model = None
        self._processor = None

    @property
    def model_id(self) -> str:
        return self._model_id

    async def load(self):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_sync)

    def _load_sync(self):
        from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
        import torch

        kwargs = {"device_map": self._device}
        if self._load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

        self._processor = AutoProcessor.from_pretrained(self._model_id)
        self._model = AutoModelForVision2Seq.from_pretrained(
            self._model_id, torch_dtype=torch.float16, **kwargs
        )

    async def unload(self):
        import torch
        del self._model
        del self._processor
        self._model = None
        self._processor = None
        torch.cuda.empty_cache()

    async def infer(self, frame: Image.Image, prompt: str, system_prompt=None):
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._infer_sync, frame, prompt, system_prompt)
        # Simulate streaming by yielding the full result as one token
        # Real streaming requires TextIteratorStreamer — see extended implementation
        yield result

    def _infer_sync(self, frame, prompt, system_prompt):
        import torch
        from transformers import TextStreamer

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": frame},
                {"type": "text", "text": prompt}
            ]
        })

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(text=text, images=[frame], return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self._model.generate(**inputs, max_new_tokens=512)

        decoded = self._processor.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        return decoded

    async def infer_multi(self, frames, prompt, system_prompt=None):
        # Qwen2-VL and InternVL2 support multiple images natively
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._infer_multi_sync, frames, prompt, system_prompt
        )
        yield result

    @property
    def supports_multi_frame(self) -> bool:
        return True
```

### 6.4 MLXAdapter (Apple Silicon)

```python
# localvisionai/adapters/mlx_adapter.py

import asyncio
from PIL import Image
from .base import AbstractModelAdapter

class MLXAdapter(AbstractModelAdapter):
    """Apple Silicon MLX backend via mlx-vlm."""

    def __init__(self, model_path: str):
        self._model_path = model_path
        self._model = None
        self._processor = None
        self._config = None

    @property
    def model_id(self) -> str:
        return self._model_path

    async def load(self):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_sync)

    def _load_sync(self):
        from mlx_vlm import load, load_config
        from mlx_vlm.prompt_utils import apply_chat_template
        self._model, self._processor = load(self._model_path)
        self._config = load_config(self._model_path)

    async def unload(self):
        import mlx.core as mx
        del self._model
        del self._processor
        mx.metal.clear_cache()

    async def infer(self, frame: Image.Image, prompt: str, system_prompt=None):
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._infer_sync, frame, prompt)
        yield result

    def _infer_sync(self, frame, prompt):
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        formatted = apply_chat_template(
            self._processor, self._config, prompt, num_images=1
        )
        return generate(
            self._model, self._processor, formatted, [frame],
            max_tokens=512, verbose=False
        )
```

### 6.5 Adapter Registry

```python
# localvisionai/adapters/registry.py

from .ollama_adapter import OllamaAdapter
from .transformers_adapter import HuggingFaceAdapter
from .llamacpp_adapter import LlamaCppAdapter
from .mlx_adapter import MLXAdapter
from .vllm_adapter import VLLMAdapter

REGISTRY = {
    "ollama": OllamaAdapter,
    "transformers": HuggingFaceAdapter,
    "llamacpp": LlamaCppAdapter,
    "mlx": MLXAdapter,
    "vllm": VLLMAdapter,
}

def get_adapter(backend: str, **kwargs):
    if backend not in REGISTRY:
        raise ValueError(
            f"Unknown backend '{backend}'. "
            f"Available: {list(REGISTRY.keys())}"
        )
    return REGISTRY[backend](**kwargs)
```

---

## 7. API Design

### 7.1 FastAPI Server

The API server runs independently from the pipeline. Jobs are submitted via REST and results stream via WebSocket.

```
Base URL: http://localhost:8765/api/v1
```

### 7.2 REST Endpoints

#### Submit a job

```
POST /api/v1/jobs
Content-Type: application/json

{
  "source": {
    "type": "file",                   // "file" | "webcam" | "rtsp" | "url" | "screen"
    "path": "/path/to/video.mp4"
  },
  "model": {
    "backend": "ollama",
    "model_id": "gemma3",
    "multi_frame": false
  },
  "sampling": {
    "strategy": "uniform",            // "uniform" | "scene" | "keyframe" | "adaptive"
    "fps": 1.0,
    "scene_threshold": 30.0
  },
  "prompt": {
    "user": "Describe what is happening in this frame.",
    "system": "You are a concise video analyst.",
    "context_mode": "sliding_window", // "none" | "sliding_window"
    "context_tokens": 200
  },
  "output": {
    "formats": ["json", "srt"],
    "output_dir": "/tmp/localvisionai/output/"
  }
}

Response 202 Accepted:
{
  "job_id": "j_abc123",
  "status": "queued",
  "created_at": "2026-04-07T12:00:00Z",
  "ws_url": "ws://localhost:8765/api/v1/jobs/j_abc123/stream"
}
```

#### Get job status

```
GET /api/v1/jobs/{job_id}

Response 200:
{
  "job_id": "j_abc123",
  "status": "running",               // "queued" | "running" | "done" | "failed"
  "progress": {
    "frames_processed": 42,
    "frames_total": 300,
    "elapsed_seconds": 86,
    "estimated_remaining_seconds": 513
  },
  "created_at": "2026-04-07T12:00:00Z",
  "started_at": "2026-04-07T12:00:01Z"
}
```

#### Get job results

```
GET /api/v1/jobs/{job_id}/results?format=json&from_ts=0&to_ts=60

Response 200:
{
  "job_id": "j_abc123",
  "results": [
    {
      "timestamp": 0.0,
      "description": "A person sits down at a desk.",
      "model_id": "gemma3",
      "backend": "ollama",
      "latency_ms": 1240,
      "token_count": 11
    },
    ...
  ]
}
```

#### List available models

```
GET /api/v1/models

Response 200:
{
  "ollama": ["gemma3", "qwen2-vl:7b", "llava:13b"],
  "transformers": [],
  "mlx": []
}
```

#### Cancel a job

```
DELETE /api/v1/jobs/{job_id}

Response 200: { "cancelled": true }
```

### 7.3 WebSocket Streaming

```
WS /api/v1/jobs/{job_id}/stream
```

The server pushes one JSON message per token as it streams from the model:

```json
// Token event
{"event": "token", "timestamp": 12.5, "token": "A "}
{"event": "token", "timestamp": 12.5, "token": "person"}

// Frame complete event
{"event": "frame_done", "timestamp": 12.5, "description": "A person walks in.", "latency_ms": 1100}

// Job complete event
{"event": "job_done", "job_id": "j_abc123", "total_frames": 300}

// Error event
{"event": "error", "message": "Model inference failed on frame at 45.0s", "timestamp": 45.0}
```

### 7.4 Pydantic Schemas

```python
# localvisionai/api/schemas.py

from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime

class SourceConfig(BaseModel):
    type: Literal["file", "webcam", "rtsp", "url", "screen"]
    path: str | None = None
    device_index: int = 0
    rtsp_url: str | None = None

class ModelConfig(BaseModel):
    backend: Literal["ollama", "transformers", "llamacpp", "mlx", "vllm"]
    model_id: str
    multi_frame: bool = False
    multi_frame_count: int = 4
    load_in_4bit: bool = False

class SamplingConfig(BaseModel):
    strategy: Literal["uniform", "scene", "keyframe", "adaptive"] = "uniform"
    fps: float = Field(1.0, gt=0, le=30)
    scene_threshold: float = Field(30.0, gt=0)
    min_interval: float = 0.5

class PromptConfig(BaseModel):
    user: str = "Describe what is happening in this frame in one sentence."
    system: str | None = None
    context_mode: Literal["none", "sliding_window"] = "none"
    context_tokens: int = 200

class OutputConfig(BaseModel):
    formats: list[Literal["console", "json", "srt", "csv", "sqlite"]] = ["console", "json"]
    output_dir: str = "/tmp/localvisionai/"

class JobRequest(BaseModel):
    source: SourceConfig
    model: ModelConfig
    sampling: SamplingConfig = SamplingConfig()
    prompt: PromptConfig = PromptConfig()
    output: OutputConfig = OutputConfig()
```

---

## 8. CLI Design

Built with Typer. All CLI options map directly to the Pydantic config models.

```bash
# Basic usage
localvisionai run --video path/to/video.mp4 --model gemma3

# With options
localvisionai run \
  --video path/to/video.mp4 \
  --backend ollama \
  --model gemma3 \
  --fps 1.0 \
  --sampler scene \
  --prompt "What objects are visible in this frame?" \
  --output-formats json srt \
  --output-dir ./results/

# Live webcam
localvisionai run --source webcam --device 0 --model qwen2-vl:7b

# RTSP stream
localvisionai run --source rtsp --rtsp-url rtsp://192.168.1.100/stream --model llava:13b

# YouTube URL
localvisionai run --source url --url https://youtube.com/watch?v=... --model gemma3 --fps 0.5

# Start the API server
localvisionai serve --host 127.0.0.1 --port 8765

# Search a processed video's timeline
localvisionai search --db results/timeline.db --query "person entering room"

# List installed models per backend
localvisionai models list

# Pull a model via Ollama
localvisionai models pull gemma3

# Benchmark a model on a test video
localvisionai benchmark --model gemma3 --video tests/fixtures/sample_video.mp4
```

### 8.1 CLI Module Structure

```python
# localvisionai/cli.py

import typer
from typing import Optional
from .pipeline import Pipeline
from .config import PipelineConfig

app = typer.Typer(name="localvisionai", help="Local AI video understanding pipeline.")

@app.command()
def run(
    video: Optional[str] = typer.Option(None, help="Path to video file"),
    source: str = typer.Option("file", help="Source type: file|webcam|rtsp|url|screen"),
    backend: str = typer.Option("ollama", help="Model backend"),
    model: str = typer.Option("gemma3", help="Model identifier"),
    fps: float = typer.Option(1.0, help="Frames per second to sample"),
    sampler: str = typer.Option("uniform", help="Sampling strategy"),
    prompt: str = typer.Option("Describe this frame.", help="Prompt sent to the model"),
    output_formats: list[str] = typer.Option(["console", "json"], help="Output formats"),
    output_dir: str = typer.Option("./output/", help="Output directory"),
    config_file: Optional[str] = typer.Option(None, help="YAML config file path"),
):
    config = PipelineConfig.from_cli(locals())
    import asyncio
    asyncio.run(Pipeline(config).run())

@app.command()
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8765),
):
    import uvicorn
    from .api.server import app as fastapi_app
    uvicorn.run(fastapi_app, host=host, port=port)

if __name__ == "__main__":
    app()
```

---

## 9. Configuration System

All configuration is expressed via Pydantic models and can be loaded from three sources, merged in priority order (highest wins):

```
CLI flags > Environment variables > YAML config file > Defaults
```

### 9.1 YAML Config Example

```yaml
# configs/default.yaml

source:
  type: file
  path: null

model:
  backend: ollama
  model_id: gemma3
  multi_frame: false
  multi_frame_count: 4

sampling:
  strategy: uniform
  fps: 1.0
  scene_threshold: 30.0

prompt:
  user: "Describe what is happening in this frame in one sentence."
  system: null
  context_mode: none
  context_tokens: 200

output:
  formats:
    - console
    - json
  output_dir: ./output/

pipeline:
  queue_size: 8         # Max frames buffered before dropping
  drop_policy: oldest   # oldest | newest | none
  retry_on_error: true
  max_retries: 3

api:
  host: 127.0.0.1
  port: 8765
  cors_origins: []
```

### 9.2 Environment Variable Overrides

All config keys are available as environment variables with the `LVA_` prefix:

```bash
export LVA_MODEL_BACKEND=transformers
export LVA_MODEL_MODEL_ID=Qwen/Qwen2-VL-7B-Instruct
export LVA_SAMPLING_FPS=0.5
export LVA_PIPELINE_QUEUE_SIZE=4
```

---

## 10. Storage & Persistence

### 10.1 SQLite Timeline Schema

```sql
-- Primary timeline table
CREATE TABLE frames (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      TEXT NOT NULL,
    timestamp   REAL NOT NULL,
    description TEXT NOT NULL,
    model_id    TEXT NOT NULL,
    backend     TEXT NOT NULL,
    latency_ms  REAL,
    token_count INTEGER,
    created_at  TEXT DEFAULT (datetime('now'))
);

-- Full-text search virtual table
CREATE VIRTUAL TABLE frames_fts USING fts5(
    description,
    content='frames',
    content_rowid='id'
);

-- Jobs table
CREATE TABLE jobs (
    id          TEXT PRIMARY KEY,
    source_path TEXT,
    config_json TEXT,
    status      TEXT DEFAULT 'queued',
    created_at  TEXT DEFAULT (datetime('now')),
    started_at  TEXT,
    finished_at TEXT,
    error       TEXT
);

-- Index for timestamp-range queries
CREATE INDEX idx_frames_job_ts ON frames (job_id, timestamp);
```

### 10.2 SRT Export Format

```
1
00:00:00,000 --> 00:00:01,000
A person walks into the room carrying a bag.

2
00:00:02,000 --> 00:00:03,000
They place the bag on a desk and sit down.
```

### 10.3 JSON Export Format

```json
{
  "job_id": "j_abc123",
  "source": "video.mp4",
  "model": "gemma3",
  "backend": "ollama",
  "generated_at": "2026-04-07T12:05:30Z",
  "frames": [
    {
      "timestamp": 0.0,
      "description": "A person walks into the room carrying a bag.",
      "latency_ms": 1240,
      "token_count": 11
    }
  ]
}
```

---

## 11. Async Pipeline Design

### 11.1 Queue Drop Policies

The bounded queue implements three drop policies when full:

```python
class DropPolicy(str, Enum):
    OLDEST = "oldest"    # Drop the oldest frame in the queue (default)
    NEWEST = "newest"    # Drop the incoming frame (newest)
    NONE = "none"        # Block the producer (no dropping — risk of lag)
```

`OLDEST` is the default. It ensures the model is always processing the most recent frames, which is critical for real-time streams.

### 11.2 Incremental Output on Long Jobs

For videos longer than 60 seconds, output handlers flush to disk every 30 seconds so that progress is not lost if the process is interrupted:

```python
class JSONOutput(AbstractOutputHandler):
    def __init__(self, path: str, flush_every: int = 30):
        self._buffer = []
        self._flush_every = flush_every  # seconds
        self._last_flush = time.time()

    async def handle(self, result: InferenceResult):
        self._buffer.append(result)
        if time.time() - self._last_flush >= self._flush_every:
            await self._flush_to_disk()
            self._last_flush = time.time()

    async def _flush_to_disk(self):
        # Atomic write via temp file + rename
        ...
```

### 11.3 RTSP Reconnection

```python
class RTSPSource(AbstractVideoSource):
    MAX_RETRIES = 5
    RETRY_DELAY = 3.0  # seconds

    async def stream(self):
        retries = 0
        while retries < self.MAX_RETRIES:
            try:
                async for frame, ts in self._stream_inner():
                    retries = 0  # reset on success
                    yield frame, ts
                break  # clean end of stream
            except av.AVError as e:
                retries += 1
                logger.warning(f"RTSP disconnect ({e}). Retry {retries}/{self.MAX_RETRIES}...")
                await asyncio.sleep(self.RETRY_DELAY)
        if retries >= self.MAX_RETRIES:
            raise RuntimeError("RTSP stream failed after max retries.")
```

---

## 12. Hardware Abstraction

### 12.1 Hardware Detection

```python
# localvisionai/utils/hardware.py

import platform
import sys

def detect_hardware() -> dict:
    info = {
        "platform": platform.system(),
        "cuda": False,
        "mps": False,       # Apple Silicon via PyTorch
        "mlx": False,       # Apple Silicon via MLX
        "cpu_cores": ...,
        "ram_gb": ...,
    }

    try:
        import torch
        info["cuda"] = torch.cuda.is_available()
        info["mps"] = torch.backends.mps.is_available()
        if info["cuda"]:
            info["vram_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    except ImportError:
        pass

    try:
        import mlx.core as mx
        info["mlx"] = True
    except ImportError:
        pass

    return info

def recommend_backend() -> str:
    hw = detect_hardware()
    if hw["mlx"]:
        return "mlx"
    if hw["cuda"]:
        return "ollama"      # Ollama handles CUDA management automatically
    return "llamacpp"        # Best CPU-only option
```

### 12.2 Auto-FPS Scaling on CPU

When running CPU-only, the pipeline measures actual inference latency after the first frame and auto-adjusts the sampler's interval to match:

```python
async def consumer():
    ...
    async for result in adapter.infer(frame, prompt):
        ...
    actual_latency = result.latency_ms / 1000

    # If we're slower than our target FPS, back off
    if actual_latency > sampler.interval:
        new_interval = actual_latency * 1.1  # 10% headroom
        sampler.interval = new_interval
        logger.info(f"Auto-adjusted FPS to {1/new_interval:.2f} based on model latency")
```

---

## 13. Plugin System

### 13.1 AbstractPlugin

```python
# localvisionai/plugins/base.py

from abc import ABC, abstractmethod
from ..adapters.base import InferenceResult

class AbstractPlugin(ABC):
    """
    Plugins receive every InferenceResult and can trigger side effects.
    They must not block — use asyncio for any I/O.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    async def on_result(self, result: InferenceResult) -> None:
        ...

    async def on_start(self, job_id: str) -> None:
        """Called when a job starts. Override for setup."""
        pass

    async def on_finish(self, job_id: str) -> None:
        """Called when a job finishes. Override for cleanup."""
        pass
```

### 13.2 Webhook Plugin

```python
# localvisionai/plugins/builtin/webhook_plugin.py

import aiohttp
from ..base import AbstractPlugin

class WebhookPlugin(AbstractPlugin):
    def __init__(self, url: str, trigger_phrase: str | None = None):
        self.url = url
        self.trigger_phrase = trigger_phrase  # Only fire if description contains this

    @property
    def name(self):
        return "webhook"

    async def on_result(self, result):
        if self.trigger_phrase and self.trigger_phrase not in result.description:
            return
        payload = {
            "timestamp": result.timestamp,
            "description": result.description,
            "model": result.model_id,
        }
        async with aiohttp.ClientSession() as session:
            await session.post(self.url, json=payload)
```

### 13.3 Plugin Discovery

Plugins are loaded from the `localvisionai.plugins` entry point group, enabling third-party packages to register themselves:

```toml
# Third-party plugin's pyproject.toml
[project.entry-points."localvisionai.plugins"]
home_assistant = "my_lva_plugin:HomeAssistantPlugin"
```

```python
# localvisionai/plugins/loader.py

import importlib.metadata

def load_plugins(config: list[dict]) -> list:
    plugins = []
    eps = importlib.metadata.entry_points(group="localvisionai.plugins")
    available = {ep.name: ep for ep in eps}
    for plugin_config in config:
        name = plugin_config["name"]
        if name in available:
            cls = available[name].load()
            plugins.append(cls(**plugin_config.get("options", {})))
    return plugins
```

---

## 14. Error Handling & Resilience

### 14.1 Error Hierarchy

```python
class LocalVisionAIError(Exception): pass
class ModelNotFoundError(LocalVisionAIError): pass
class ModelInferenceError(LocalVisionAIError): pass
class SourceOpenError(LocalVisionAIError): pass
class SourceReadError(LocalVisionAIError): pass
class ConfigValidationError(LocalVisionAIError): pass
```

### 14.2 Per-Frame Error Recovery

Errors on individual frames never crash the pipeline. They are logged, and the frame is skipped:

```python
async def consumer():
    while True:
        frame, ts = await queue.get()
        try:
            tokens = []
            async for token in adapter.infer(frame, prompt):
                tokens.append(token)
            result = InferenceResult(...)
            for handler in handlers:
                await handler.handle(result)
        except ModelInferenceError as e:
            logger.warning(f"Inference failed at {ts:.2f}s: {e}. Skipping frame.")
        except Exception as e:
            logger.error(f"Unexpected error at {ts:.2f}s: {e}", exc_info=True)
```

### 14.3 Structured Logging

```python
# localvisionai/utils/logging.py

import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "time": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "component": record.name,
            "message": record.getMessage(),
            **({"exc": self.formatException(record.exc_info)} if record.exc_info else {})
        })
```

---

## 15. Testing Strategy

### 15.1 Unit Tests

Every component is tested in isolation using `pytest` and `pytest-asyncio`. The model adapter is mocked to avoid requiring real GPU hardware in CI.

```python
# tests/unit/test_sampler.py

import pytest
from PIL import Image
from localvisionai.sampling.uniform_sampler import UniformSampler

def make_frame():
    return Image.new("RGB", (224, 224))

def test_uniform_sampler_1fps():
    sampler = UniformSampler(fps=1.0)
    assert sampler.should_process(make_frame(), 0.0) == True
    assert sampler.should_process(make_frame(), 0.5) == False
    assert sampler.should_process(make_frame(), 1.0) == True
```

```python
# tests/unit/test_pipeline.py

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from localvisionai.pipeline import Pipeline

@pytest.mark.asyncio
async def test_pipeline_processes_frames():
    mock_adapter = AsyncMock()
    mock_adapter.infer.return_value = async_gen(["A ", "person ", "sits."])
    # ... test pipeline processes frames and calls adapter
```

### 15.2 Integration Tests

Integration tests run against a real Ollama instance and a bundled 10-second test video. These are marked `@pytest.mark.integration` and excluded from default CI runs.

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_ollama_adapter_real_inference():
    from localvisionai.adapters.ollama_adapter import OllamaAdapter
    adapter = OllamaAdapter(model="gemma3")
    await adapter.load()
    frame = Image.open("tests/fixtures/sample_frame.jpg")
    tokens = []
    async for token in adapter.infer(frame, "What is in this image?"):
        tokens.append(token)
    assert len(tokens) > 0
    assert isinstance("".join(tokens), str)
```

### 15.3 CI Pipeline (GitHub Actions)

```yaml
# .github/workflows/test.yml

name: Tests
on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[dev]"
      - run: pytest tests/unit/ -v --cov=localvisionai --cov-report=xml
      - uses: codecov/codecov-action@v3
```

---

## 16. Performance Targets & Benchmarks

### 16.1 Latency Targets

| Hardware | Model | Quantization | Target Latency |
|---|---|---|---|
| NVIDIA RTX 3080 (10GB) | Gemma 3 9B | Q4_K_M | < 1.5s/frame |
| NVIDIA RTX 3080 (10GB) | Qwen2-VL 7B | Q4_K_M | < 2.0s/frame |
| Apple M2 Pro | Gemma 3 9B | MLX 4-bit | < 2.5s/frame |
| Apple M1 | LLaVA 7B | MLX 4-bit | < 4.0s/frame |
| CPU only (8-core) | LLaVA 7B | Q4_K_M | < 15s/frame |

### 16.2 Memory Targets

| Model | VRAM (fp16) | VRAM (Q4) |
|---|---|---|
| LLaVA 7B | 14 GB | 5 GB |
| Qwen2-VL 7B | 15 GB | 5.5 GB |
| Gemma 3 9B | 18 GB | 6 GB |
| InternVL2 8B | 16 GB | 6 GB |

### 16.3 Benchmark Command

```bash
localvisionai benchmark \
  --model gemma3 \
  --backend ollama \
  --video tests/fixtures/sample_video.mp4 \
  --frames 20 \
  --report benchmark_results.json
```

---

## 17. Security Considerations

- **No external network calls during inference.** The HTTP client (`aiohttp`) is only used by the optional webhook plugin. All other operations are local.
- **API server binds to localhost by default.** Enabling external binding requires explicit `--host 0.0.0.0` flag plus documented warning.
- **No frame storage by default.** Frames are processed in memory and never written to disk unless the user explicitly enables `--save-frames`.
- **yt-dlp usage is user-initiated.** The user explicitly passes a URL; the project does not proactively download anything.
- **Input validation via Pydantic.** All API inputs are validated before any processing begins.
- **No eval(), exec(), or dynamic imports from user input.** Plugin loading uses entry points only — not arbitrary file paths.

---

## 18. Dependency Matrix

### Core (always required)

```toml
[project]
name = "localvisionai"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "av>=12.0",                  # PyAV — FFmpeg bindings
    "Pillow>=10.0",              # PIL Image
    "numpy>=1.24",               # Frame array operations
    "pydantic>=2.0",             # Config validation, API schemas
    "typer>=0.12",               # CLI
    "fastapi>=0.110",            # REST + WebSocket API
    "uvicorn>=0.29",             # ASGI server
    "aiosqlite>=0.19",           # Async SQLite
    "pyyaml>=6.0",               # Config file parsing
    "rich>=13.0",                # Terminal output formatting
]
```

### Optional extras

```toml
[project.optional-dependencies]
ollama = ["ollama>=0.2"]
transformers = [
    "transformers>=4.40",
    "accelerate>=0.28",
    "bitsandbytes>=0.43",        # 4-bit quantization
    "sentencepiece>=0.2",
    "torch>=2.2",
]
llamacpp = ["llama-cpp-python>=0.2.70"]
mlx = ["mlx>=0.12", "mlx-vlm>=0.1"]
vllm = ["vllm>=0.4"]
screen = ["mss>=9.0"]
url = ["yt-dlp>=2024.1"]
scene = ["scenedetect>=0.6"]
ocr = ["pytesseract>=0.3", "tesseract"]
whisper = ["openai-whisper>=20231117"]
rag = ["chromadb>=0.5", "sentence-transformers>=2.7"]
gradio = ["gradio>=4.0"]
web = ["aiohttp>=3.9"]           # For webhook plugin
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-cov>=5.0",
    "ruff>=0.4",
    "mypy>=1.10",
    "httpx>=0.27",               # FastAPI test client
]
all = [
    "localvisionai[ollama,transformers,llamacpp,mlx,screen,url,scene,ocr,whisper,rag,gradio,web]"
]
```

---

## 19. Environment & Installation

```bash
# Minimal install (Ollama backend only)
pip install localvisionai[ollama]

# With HuggingFace transformers
pip install localvisionai[transformers]

# Apple Silicon full install
pip install localvisionai[ollama,mlx,screen,url,scene]

# Full install (everything)
pip install localvisionai[all]

# Development install
git clone https://github.com/localvisionai/localvisionai
cd localvisionai
pip install -e ".[all,dev]"
pytest tests/unit/
```

### System dependencies

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg tesseract-ocr

# macOS
brew install ffmpeg tesseract

# Windows
# FFmpeg: Download from ffmpeg.org and add to PATH
# Tesseract: Download installer from UB-Mannheim GitHub
```

---

## 20. Contribution Guide for Adapters

Adding a new model backend requires three things:

**1. Create the adapter file**

```python
# localvisionai/adapters/my_backend_adapter.py

from .base import AbstractModelAdapter

class MyBackendAdapter(AbstractModelAdapter):
    # Implement all abstract methods
    ...
```

**2. Register it in the registry**

```python
# localvisionai/adapters/registry.py

from .my_backend_adapter import MyBackendAdapter
REGISTRY["my_backend"] = MyBackendAdapter
```

**3. Add tests**

```python
# tests/unit/test_my_backend_adapter.py

@pytest.mark.asyncio
async def test_my_adapter_streams_tokens():
    adapter = MyBackendAdapter(model_id="test-model")
    frame = Image.new("RGB", (224, 224))
    tokens = []
    async for token in adapter.infer(frame, "Describe this."):
        tokens.append(token)
    assert len(tokens) > 0
```

**4. Document supported models in `docs/model-adapters.md`**

That's it. Open a pull request and the CI will validate the adapter automatically.

---

*Document maintained by the LocalVisionAI core team. API and schema specifications are normative. Implementation details are guidance. For questions, open a GitHub Discussion.*
