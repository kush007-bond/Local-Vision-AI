# Audio Input Feature — Architecture & Implementation Plan

## Overview

This document describes how to extend LocalVisionAI to support **audio analysis alongside video frames**. When enabled, the pipeline extracts audio segments that are time-aligned to each sampled frame and passes them to the model so it can reason about both modalities together.

Two modes are supported:

| Mode | How it works | Best for |
|---|---|---|
| **native** | Raw audio bytes sent directly to the model | OpenAI GPT-4o-audio, Gemini 2.0 Flash — models with native audio support |
| **transcribe** | Audio is transcribed to text (Whisper), transcript injected into the prompt | All other backends (Ollama, Anthropic, HuggingFace, LM Studio, etc.) |
| **auto** *(default)* | Uses `native` if the adapter declares `supports_audio = True`, falls back to `transcribe` | Recommended — requires no manual config |

---

## Design Principles

1. **Zero breaking changes** — existing configs without `audio:` section work exactly as before.
2. **Audio runs alongside, not replacing, video** — frames are still the primary signal; audio is injected as additional context.
3. **Adapter-driven capability** — adapters declare what they support; the pipeline adapts automatically.
4. **Efficiency first** — audio is extracted once via `ffmpeg` subprocess into a RAM buffer, segmented lazily, and transcribed only when needed using a cached Whisper model.

---

## Component Map

```
localvisionai/
├── audio/                          ← NEW package
│   ├── __init__.py
│   ├── base.py                     ← AudioChunk dataclass + AbstractAudioExtractor
│   ├── ffmpeg_extractor.py         ← Extracts audio from file/RTSP via ffmpeg subprocess
│   ├── transcriber.py              ← WhisperTranscriber (wraps faster-whisper or openai-whisper)
│   └── segmenter.py                ← Slices the full audio buffer to per-frame windows
│
├── adapters/
│   ├── base.py                     ← AbstractModelAdapter gains supports_audio + infer_with_audio()
│   ├── openai_adapter.py           ← Override supports_audio=True, infer_with_audio() sends audio
│   ├── gemini_adapter.py           ← Override supports_audio=True, infer_with_audio() sends audio
│   └── (all others unchanged)      ← Use transcription fallback automatically
│
├── config.py                       ← AudioConfig added to PipelineConfig
├── pipeline.py                     ← _producer yields optional audio; _consumer routes to infer/infer_with_audio
└── prompts/
    └── builder.py                  ← build_prompt() accepts optional transcript kwarg
```

---

## 1. New Data Types — `localvisionai/audio/base.py`

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class AudioChunk:
    """Time-aligned audio segment corresponding to one sampled frame."""
    data: bytes           # Raw PCM or AAC bytes (format depends on extractor)
    sample_rate: int      # e.g. 16000 for Whisper-compatible PCM
    channels: int         # 1 (mono) recommended for Whisper
    start_ts: float       # Start of the audio window (seconds)
    end_ts: float         # End of the audio window (seconds)
    transcript: Optional[str] = None  # Populated by WhisperTranscriber if used
```

---

## 2. Configuration — `AudioConfig` added to `localvisionai/config.py`

```python
class AudioConfig(BaseModel):
    enabled: bool = False
    mode: Literal["native", "transcribe", "auto"] = "auto"
    
    # How many seconds of audio to send per frame
    window_seconds: float = Field(3.0, gt=0, le=30.0)
    
    # Whisper settings (transcribe mode only)
    whisper_model: str = "base"          # tiny / base / small / medium / large
    whisper_device: str = "auto"         # auto | cpu | cuda
    whisper_language: Optional[str] = None  # None = auto-detect
    
    # ffmpeg extraction quality
    sample_rate: int = 16000             # Hz — 16000 is optimal for Whisper
    channels: int = 1                    # Mono is sufficient for speech
```

`PipelineConfig` gains a new field:
```python
class PipelineConfig(BaseModel):
    ...
    audio: AudioConfig = AudioConfig()   # disabled by default
```

---

## 3. Audio Extraction — `localvisionai/audio/ffmpeg_extractor.py`

The extractor runs **once** at pipeline start, writing the entire audio track to a `numpy` array in memory (or a temp WAV file for large videos). It is I/O-bound so it runs in `asyncio.to_thread` to avoid blocking the event loop.

```
ffmpeg -i input.mp4 -vn -ar 16000 -ac 1 -f f32le pipe:1
```

The raw float32 samples are stored in a `numpy` array indexed by sample position. Given a timestamp `ts` and `window_seconds`, the segment is a simple array slice — O(1) cost per frame.

### Handling live sources (RTSP / Webcam)

For live sources, a background thread continuously reads ffmpeg stdout into a **circular ring buffer** of configurable size (default: 60 seconds). The segmenter reads from the ring buffer using the frame timestamp to align the window.

---

## 4. Segmenter — `localvisionai/audio/segmenter.py`

```python
class AudioSegmenter:
    """Slices the extracted audio buffer into per-frame AudioChunks."""

    def __init__(self, samples: np.ndarray, sample_rate: int, channels: int): ...

    def get_chunk(self, frame_ts: float, window_seconds: float) -> AudioChunk:
        """
        Return the audio window ending at frame_ts.
        Window = [frame_ts - window_seconds, frame_ts]
        Clamps to [0, duration] at boundaries.
        """
        ...
```

---

## 5. Transcriber — `localvisionai/audio/transcriber.py`

```python
class WhisperTranscriber:
    """
    Singleton wrapper around faster-whisper (preferred) or openai-whisper.
    Loaded once at pipeline start, reused for every frame.
    Transcription runs in asyncio.to_thread (CPU-bound).
    """

    def __init__(self, model_size: str, device: str, language: Optional[str]): ...

    async def transcribe(self, chunk: AudioChunk) -> str:
        """Returns transcript string. Caches empty-audio detection to skip silences."""
        ...
```

**Dependency**: `faster-whisper` (preferred — 4× faster than `openai-whisper`, same accuracy).

---

## 6. Adapter Interface Changes — `localvisionai/adapters/base.py`

Two additions to `AbstractModelAdapter`:

```python
@property
def supports_audio(self) -> bool:
    """True if this adapter can accept raw audio bytes alongside a frame."""
    return False

async def infer_with_audio(
    self,
    frame: Image.Image,
    audio: AudioChunk,
    prompt: str,
    system_prompt: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    Default implementation: transcribes audio and injects transcript into prompt,
    then falls back to regular infer(). Adapters with native audio override this.
    """
    # Base default — should not be called; pipeline dispatches correctly.
    raise NotImplementedError
```

No existing adapters need to change unless they want native audio support.

---

## 7. Native Audio Adapters

### OpenAI (`openai_adapter.py`)

```python
@property
def supports_audio(self) -> bool:
    # Enable for audio-capable models
    return "audio" in self._model_id or self._model_id.startswith("gpt-4o")

async def infer_with_audio(self, frame, audio, prompt, system_prompt=None):
    # Encode frame as base64 image
    # Encode audio as base64 WAV
    # Build message with content=[image_url, audio_url, text]
    # Use client.chat.completions.create() streaming
```

OpenAI audio input format: `{"type": "input_audio", "input_audio": {"data": "<base64>", "format": "wav"}}`

### Gemini (`gemini_adapter.py`)

```python
@property
def supports_audio(self) -> bool:
    return True  # All Gemini 2.0+ models support audio

async def infer_with_audio(self, frame, audio, prompt, system_prompt=None):
    # Gemini accepts audio via inline_data with mime_type "audio/wav"
    # Build Part list: [image_part, audio_part, text_part]
    # Use generate_content() streaming
```

---

## 8. Pipeline Changes — `localvisionai/pipeline.py`

### `run()` additions

```python
# After building source and sampler:
audio_extractor = None
audio_segmenter = None
transcriber = None

if self.config.audio.enabled:
    audio_extractor = FfmpegAudioExtractor(self.config)
    raw_samples = await asyncio.to_thread(audio_extractor.extract)
    audio_segmenter = AudioSegmenter(raw_samples, self.config.audio.sample_rate, ...)
    
    if self._use_transcription(adapter):
        transcriber = WhisperTranscriber(
            model_size=self.config.audio.whisper_model,
            device=self.config.audio.whisper_device,
            language=self.config.audio.whisper_language,
        )
```

### `_producer()` — unchanged

The producer continues yielding `(frame, ts)`. Audio is extracted in the consumer from the pre-loaded buffer using `ts` as a lookup key — no change to the queue protocol.

### `_consumer()` — audio routing

```python
# Inside the per-frame inference block:
audio_chunk = None
if audio_segmenter:
    audio_chunk = audio_segmenter.get_chunk(ts, self.config.audio.window_seconds)

if audio_chunk and adapter.supports_audio and use_native_mode:
    # Native path: send frame + raw audio to model
    async for token in adapter.infer_with_audio(frame, audio_chunk, user_prompt, system_prompt):
        tokens.append(token)

elif audio_chunk and transcriber:
    # Transcription path: transcribe, then inject into prompt
    audio_chunk.transcript = await transcriber.transcribe(audio_chunk)
    enriched_prompt = build_prompt(self.config.prompt, ..., transcript=audio_chunk.transcript)
    async for token in adapter.infer(frame, enriched_prompt, system_prompt):
        tokens.append(token)

else:
    # No audio — existing path unchanged
    async for token in adapter.infer(frame, user_prompt, system_prompt):
        tokens.append(token)
```

### Helper

```python
def _use_transcription(self, adapter) -> bool:
    mode = self.config.audio.mode
    if mode == "transcribe":
        return True
    if mode == "native":
        return False
    # auto: use native if supported, else transcribe
    return not adapter.supports_audio
```

---

## 9. Prompt Builder Changes — `localvisionai/prompts/builder.py`

`build_prompt()` gains an optional `transcript` argument:

```python
def build_prompt(
    config: PromptConfig,
    context_summary: str = "",
    transcript: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    user = config.user
    if transcript:
        user = f"{user}\n\n[Audio transcript for this segment: \"{transcript}\"]"
    ...
```

---

## 10. `InferenceResult` changes — `localvisionai/adapters/base.py`

```python
@dataclass
class InferenceResult:
    ...
    audio_transcript: Optional[str] = None   # Populated when transcription used
    audio_mode: Optional[str] = None         # "native" | "transcribe" | None
```

---

## 11. CLI changes — `localvisionai/cli.py`

New flags added to the `run` command:

```
--audio / --no-audio          Enable audio analysis (default: off)
--audio-mode [auto|native|transcribe]
--audio-window FLOAT          Seconds of audio per frame (default: 3.0)
--whisper-model TEXT          Whisper model size (default: base)
```

Example:
```bash
localvisionai run --video interview.mp4 --backend openai --model gpt-4o --audio
localvisionai run --video lecture.mp4 --backend ollama --model llava --audio --whisper-model small
```

---

## 12. Optional Dependencies — `pyproject.toml`

```toml
[project.optional-dependencies]
audio = [
    "faster-whisper>=1.0",   # Preferred transcription backend
    "numpy>=1.24",           # Audio buffer manipulation
    # ffmpeg is a system dependency — checked at runtime with a helpful error
]

# Include audio in [all]
all = [..., "localvisionai[audio]"]
```

`ffmpeg` must be on `PATH` (or configured via `LVA_FFMPEG_PATH`). The extractor checks for it at startup and raises a clear `ImportError`-style message if missing.

---

## 13. Execution Flow (End-to-End)

```
pipeline.run()
│
├── FfmpegAudioExtractor.extract()          # ffmpeg subprocess → numpy array (async thread)
├── WhisperTranscriber.load()               # Load Whisper once (if transcribe mode)
│
├── _producer()                             # Unchanged — yields (frame, ts) tuples
│
└── _consumer() [per frame]
    │
    ├── AudioSegmenter.get_chunk(ts)        # O(1) array slice
    │
    ├── [native path]  adapter.infer_with_audio(frame, chunk, prompt)
    │       └── OpenAI / Gemini — frame + audio bytes in single API call
    │
    └── [transcribe path]
            ├── WhisperTranscriber.transcribe(chunk)   # async thread
            ├── build_prompt(..., transcript=text)
            └── adapter.infer(frame, enriched_prompt)  # existing path
```

---

## 14. Backend Compatibility Matrix

| Backend | Native Audio | Transcription Fallback | Notes |
|---|---|---|---|
| OpenAI (`gpt-4o`, `gpt-4o-audio-preview`) | ✅ | ✅ | Send WAV as base64 in messages |
| Gemini (`gemini-2.0-flash`, `gemini-1.5-pro`) | ✅ | ✅ | Inline audio data supported |
| Anthropic / Claude | ❌ | ✅ | No audio API — transcript injected into text |
| Ollama (LLaVA, Gemma3, etc.) | ❌ | ✅ | Transcript injected into text |
| HuggingFace Transformers | ❌ | ✅ | Transcript injected into text |
| LM Studio | ❌ | ✅ | Transcript injected into text |
| llama.cpp | ❌ | ✅ | Transcript injected into text |
| MLX | ❌ | ✅ | Transcript injected into text |

---

## 15. Implementation Order (Phases)

### Phase 1 — Core Audio Infrastructure
1. Create `localvisionai/audio/` package with `AudioChunk`, `FfmpegAudioExtractor`, `AudioSegmenter`
2. Add `AudioConfig` to `config.py` and `PipelineConfig`
3. Add `supports_audio` property and `infer_with_audio()` stub to `AbstractModelAdapter`
4. Modify `pipeline.py` to extract audio at startup and route per-frame

### Phase 2 — Transcription Path
5. Implement `WhisperTranscriber` with `faster-whisper`
6. Modify `build_prompt()` to accept and inject transcripts
7. Wire transcription into `_consumer()` fallback path
8. Add unit tests: mock Whisper, mock ffmpeg output

### Phase 3 — Native Audio Adapters
9. Implement `infer_with_audio()` in `OpenAIAdapter`
10. Implement `infer_with_audio()` in `GeminiAdapter`
11. Integration tests for each native adapter

### Phase 4 — CLI & Config
12. Add `--audio*` flags to `cli.py`
13. Map flags through `PipelineConfig.from_cli()`
14. Add `audio_transcript` and `audio_mode` fields to `InferenceResult`
15. Update `JSONOutput` to include audio metadata

### Phase 5 — Frontend & API
16. Expose `audio_enabled` in the API `/status` response
17. Show transcript in the React UI result cards
18. Add audio toggle to the job submission form

---

## 16. Key Design Decisions & Rationale

| Decision | Rationale |
|---|---|
| **ffmpeg for extraction** | Handles every codec, container, and stream type. No Python codec library matches its breadth. |
| **Full audio pre-loaded into RAM** | For video files (not live), this allows O(1) random-access slicing per frame with zero per-frame I/O. Typical 1-hour video at 16kHz mono float32 ≈ 225 MB — acceptable. For live streams, a ring buffer is used instead. |
| **`window_seconds` centered on frame ts** | Gives the model context before and after the visual moment, catching speech that slightly precedes or follows an action. Default 3 s is a good balance between context and token cost. |
| **Singleton Whisper model** | Whisper model loading is expensive (1–4 s). Load once, reuse for every frame. |
| **`faster-whisper` over `openai-whisper`** | 4× faster transcription, lower memory, supports quantization (int8/float16). Drop-in replacement. |
| **Audio not added to `asyncio.Queue`** | Adding audio to the queue would double the queue memory for every unprocessed frame. Instead, audio is looked up from the pre-loaded buffer using `ts` at consume time — simpler and more memory-efficient. |
| **Transcript injected into prompt text** | Avoids the need for a separate "audio understanding" model for most backends. Every existing LLM can reason over transcript text effectively. |
