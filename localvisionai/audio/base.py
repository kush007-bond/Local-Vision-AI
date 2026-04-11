"""Audio data types and the abstract extractor interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class AudioChunk:
    """Time-aligned audio segment corresponding to one sampled frame.

    The `data` field carries raw PCM bytes (float32 little-endian by default,
    matching the ffmpeg extractor's output). Adapters that need a different
    format (e.g. base64 WAV for OpenAI) are expected to transcode from this
    canonical representation at send time.
    """

    data: bytes
    sample_rate: int
    channels: int
    start_ts: float
    end_ts: float
    transcript: Optional[str] = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end_ts - self.start_ts)

    @property
    def is_empty(self) -> bool:
        return len(self.data) == 0


class AbstractAudioExtractor(ABC):
    """Interface for audio extractors.

    Concrete implementations either pre-load the full audio track (file
    sources) or expose a ring buffer for live sources. The pipeline interacts
    with them exclusively through `extract()` and `close()`.
    """

    @abstractmethod
    def extract(self) -> Any:
        """Return a numpy array of samples (or raise on failure).

        For live sources, this may return the current ring-buffer view; for
        file sources, it blocks until the entire track has been read.
        """
        ...

    def close(self) -> None:
        """Release any held resources (subprocess handles, temp files)."""
        return None
