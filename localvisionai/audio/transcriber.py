"""Whisper transcription wrapper.

Prefers `faster-whisper` (≈4× faster than `openai-whisper`). The model is
loaded once, reused for every frame, and called through `asyncio.to_thread`
so the event loop stays responsive during the CPU-bound transcription.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

import numpy as np

from localvisionai.exceptions import ModelNotFoundError
from localvisionai.utils.logging import get_logger
from .base import AudioChunk
from .segmenter import AudioSegmenter

logger = get_logger(__name__)


# Silence threshold (mean absolute amplitude of float32 samples). Chunks
# below this level skip the expensive transcription call entirely.
_SILENCE_THRESHOLD = 1e-3


class WhisperTranscriber:
    """Singleton-style Whisper wrapper.

    Usage:
        transcriber = WhisperTranscriber("base", "auto", None)
        await transcriber.load()
        text = await transcriber.transcribe(chunk)
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        language: Optional[str] = None,
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._language = language
        self._model: Any = None
        self._backend: Optional[str] = None  # "faster-whisper" | "whisper"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def load(self) -> None:
        """Load the Whisper model — prefers faster-whisper, falls back to openai-whisper."""
        if self._model is not None:
            return

        await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        # Prefer faster-whisper
        try:
            from faster_whisper import WhisperModel  # type: ignore

            compute_type = "int8"
            device = self._device
            if device == "auto":
                device = "auto"  # faster-whisper understands "auto"
            self._model = WhisperModel(
                self._model_size,
                device=device,
                compute_type=compute_type,
            )
            self._backend = "faster-whisper"
            logger.info(
                f"Whisper loaded via faster-whisper — model={self._model_size} "
                f"device={device} compute_type={compute_type}"
            )
            return
        except ImportError:
            pass

        # Fall back to openai-whisper
        try:
            import whisper  # type: ignore

            device = None if self._device == "auto" else self._device
            self._model = whisper.load_model(self._model_size, device=device)
            self._backend = "whisper"
            logger.info(
                f"Whisper loaded via openai-whisper — model={self._model_size} "
                f"device={device or 'auto'}"
            )
            return
        except ImportError:
            pass

        raise ModelNotFoundError(
            "No Whisper backend installed. Install faster-whisper (preferred) "
            "with: pip install localvisionai[audio]"
        )

    async def unload(self) -> None:
        self._model = None
        self._backend = None

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    async def transcribe(self, chunk: AudioChunk) -> str:
        """Transcribe a single AudioChunk to plain text.

        Silent / empty chunks skip the model call and return "" immediately.
        """
        if self._model is None:
            raise RuntimeError("WhisperTranscriber.load() must be called before transcribe().")

        if chunk.is_empty:
            return ""

        audio = AudioSegmenter.chunk_to_numpy(chunk)
        if audio.size == 0:
            return ""

        # Cheap silence detector — saves tens of ms per silent chunk.
        if float(np.mean(np.abs(audio))) < _SILENCE_THRESHOLD:
            return ""

        # Whisper expects 16 kHz mono float32. If the source sample rate
        # differs we fall back to a simple linear resample (good enough for
        # speech; a proper polyphase filter is unnecessary for this use case).
        if chunk.sample_rate != 16000:
            audio = _linear_resample(audio, chunk.sample_rate, 16000)

        return await asyncio.to_thread(self._transcribe_sync, audio)

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        if self._backend == "faster-whisper":
            segments, _info = self._model.transcribe(
                audio,
                language=self._language,
                beam_size=1,
                vad_filter=False,
            )
            return " ".join(seg.text.strip() for seg in segments).strip()

        # openai-whisper path
        result = self._model.transcribe(
            audio,
            language=self._language,
            fp16=False,
        )
        return (result.get("text") or "").strip()


def _linear_resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Minimal linear-interpolation resampler.

    Whisper is trained on 16 kHz mono; any reasonable resample is fine. This
    avoids a hard dependency on scipy/librosa.
    """
    if src_sr == dst_sr or audio.size == 0:
        return audio
    duration = audio.size / float(src_sr)
    dst_n = int(round(duration * dst_sr))
    if dst_n <= 0:
        return np.zeros(0, dtype=np.float32)
    src_x = np.linspace(0.0, duration, num=audio.size, endpoint=False)
    dst_x = np.linspace(0.0, duration, num=dst_n, endpoint=False)
    return np.interp(dst_x, src_x, audio).astype(np.float32)
