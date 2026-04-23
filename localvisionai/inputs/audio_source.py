"""Audio-only source — produces synthetic 1×1 placeholder frames timed to
audio segments.

When the user selects source_type='audio', there is no video stream.  The
pipeline still expects (frame, timestamp) pairs, so this source emits a tiny
blank PIL image as a visual placeholder for every audio window boundary.  The
audio chunk for each window is passed natively to the model adapter (requires
an adapter with ``supports_audio = True``, e.g. Gemini, GPT-4o, or Claude).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncGenerator, Optional, Tuple

from PIL import Image

from localvisionai.exceptions import SourceOpenError, SourceReadError
from localvisionai.utils.logging import get_logger
from .base import AbstractVideoSource

logger = get_logger(__name__)

# Blank 1×1 white pixel reused for every frame emission
_BLANK_FRAME = Image.new("RGB", (1, 1), color=(255, 255, 255))


class AudioOnlySource(AbstractVideoSource):
    """
    Emits timed placeholder frames at a fixed interval to drive audio-only analysis.

    The pipeline audio segmenter slices audio around each emitted timestamp.
    The audio chunk is forwarded natively to the model adapter alongside the
    placeholder frame so the model can reason about spoken content directly.

    Args:
        path:            Path to the audio or video file (audio track is extracted).
        window_seconds:  How many seconds of audio per 'frame'. Should match
                         audio.window_seconds in PipelineConfig.
        duration:        Override total duration (seconds). If None, ffprobe is
                         used to determine it automatically.
    """

    def __init__(
        self,
        path: str,
        window_seconds: float = 3.0,
        duration: Optional[float] = None,
    ) -> None:
        self._path = path
        self._window = window_seconds
        self._duration = duration

    async def open(self) -> None:
        p = Path(self._path)
        if not p.exists():
            raise SourceOpenError(f"Audio file not found: {self._path}")

        if self._duration is None:
            self._duration = await asyncio.get_event_loop().run_in_executor(
                None, self._probe_duration
            )

        logger.info(
            f"AudioOnlySource opened: {self._path} "
            f"duration={self._duration:.1f}s window={self._window}s"
        )

    def _probe_duration(self) -> float:
        """Use ffprobe to determine file duration in seconds."""
        import shutil
        import subprocess
        import json

        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            logger.warning("ffprobe not found — defaulting to 3600s duration estimate.")
            return 3600.0

        try:
            result = subprocess.run(
                [
                    ffprobe,
                    "-v", "quiet",
                    "-print_format", "json",
                    "-show_format",
                    self._path,
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            info = json.loads(result.stdout)
            dur = float(info.get("format", {}).get("duration", 0) or 0)
            return dur if dur > 0 else 3600.0
        except Exception as e:
            logger.warning(f"ffprobe failed ({e}) — defaulting to 3600s.")
            return 3600.0

    async def close(self) -> None:
        pass  # No resources to release

    @property
    def metadata(self) -> dict:
        return {
            "duration": self._duration,
            "fps": 1.0 / max(self._window, 0.1),
            "width": 1,
            "height": 1,
            "codec": "audio-only",
        }

    async def stream(self) -> AsyncGenerator[Tuple[Image.Image, float], None]:
        if self._duration is None:
            raise SourceReadError("AudioOnlySource is not open. Call open() first.")

        ts = 0.0
        step = self._window
        while ts < self._duration:
            yield _BLANK_FRAME.copy(), ts
            ts += step
            # Yield control so the event loop stays responsive
            await asyncio.sleep(0)
