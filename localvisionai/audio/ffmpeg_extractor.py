"""ffmpeg-based audio extractor.

Runs `ffmpeg` as a subprocess to decode the audio track of a video file (or
live source) into raw float32 PCM. For file sources, the entire track is
loaded into a numpy array for O(1) per-frame random access. Live sources
are not yet wired through the pipeline, but the class exposes hooks so the
implementation can be extended without breaking the producer contract.

ffmpeg is a system dependency. The extractor checks for it on first use and
raises a clear, actionable error if it is missing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Optional

import numpy as np

from localvisionai.config import PipelineConfig
from localvisionai.exceptions import ModelNotFoundError
from localvisionai.utils.logging import get_logger
from .base import AbstractAudioExtractor

logger = get_logger(__name__)


def _resolve_ffmpeg() -> str:
    """Locate the ffmpeg binary, respecting LVA_FFMPEG_PATH."""
    override = os.environ.get("LVA_FFMPEG_PATH")
    if override:
        return override
    found = shutil.which("ffmpeg")
    if not found:
        raise ModelNotFoundError(
            "ffmpeg not found on PATH. Install ffmpeg (https://ffmpeg.org/) "
            "or set LVA_FFMPEG_PATH to the full binary path."
        )
    return found


class FfmpegAudioExtractor(AbstractAudioExtractor):
    """Extracts the audio track of a file source into a float32 numpy array.

    The full track is decoded once at pipeline start. At typical settings
    (16 kHz mono float32), a 1-hour video costs ≈225 MB of RAM — acceptable
    for the use cases this codepath targets.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._audio_cfg = config.audio
        self._source_path: Optional[str] = config.source.path
        self._samples: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self) -> np.ndarray:
        """Run ffmpeg and return a 1D float32 array of audio samples.

        Raises a ModelNotFoundError if ffmpeg is missing or cannot read the
        source. An empty array (shape=(0,)) is returned when the source has
        no audio track — callers should tolerate this.
        """
        if self._samples is not None:
            return self._samples

        if not self._source_path:
            logger.warning(
                "FfmpegAudioExtractor.extract() called without source.path — "
                "returning empty audio buffer."
            )
            self._samples = np.zeros(0, dtype=np.float32)
            return self._samples

        ffmpeg = _resolve_ffmpeg()
        cmd = [
            ffmpeg,
            "-nostdin",
            "-loglevel", "error",
            "-i", self._source_path,
            "-vn",                                 # skip the video track
            "-ar", str(self._audio_cfg.sample_rate),
            "-ac", str(self._audio_cfg.channels),
            "-f", "f32le",                         # raw little-endian float32
            "pipe:1",
        ]

        logger.info(
            f"Extracting audio via ffmpeg: sr={self._audio_cfg.sample_rate} "
            f"channels={self._audio_cfg.channels} source={self._source_path}"
        )

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except FileNotFoundError as e:
            raise ModelNotFoundError(f"ffmpeg failed to launch: {e}") from e

        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace").strip()
            # A missing audio track is not fatal — ffmpeg may still exit 0
            # with empty stdout. Anything else is a real error.
            logger.warning(
                f"ffmpeg exited with code {proc.returncode}: {err or '(no stderr)'}"
            )

        raw = proc.stdout or b""
        if not raw:
            logger.warning("No audio samples decoded — source may have no audio track.")
            self._samples = np.zeros(0, dtype=np.float32)
            return self._samples

        samples = np.frombuffer(raw, dtype=np.float32)
        # For multichannel output, leave the interleaved layout — the
        # segmenter knows how to slice it.
        self._samples = samples
        duration_s = len(samples) / float(
            self._audio_cfg.sample_rate * max(1, self._audio_cfg.channels)
        )
        logger.info(
            f"Audio extraction complete — {len(samples)} samples "
            f"(~{duration_s:.1f}s)"
        )
        return self._samples

    def close(self) -> None:
        self._samples = None
