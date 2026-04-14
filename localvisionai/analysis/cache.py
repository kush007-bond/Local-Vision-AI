"""Persistent JSON cache for offline video analysis results.

One JSON file per video stores per-frame descriptions, the full audio
transcript, and the generated summary. This allows the expensive analysis
pass to be skipped on re-runs and enables the Q&A phase to be resumed
independently.

Cache file layout:
    <output_dir>/<video_stem>_analysis.json
    <output_dir>/<video_stem>_thumbnail.jpg  (first sampled frame)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from localvisionai.adapters.base import InferenceResult

_CACHE_VERSION = 1


@dataclass
class FrameRecord:
    """One analysed video frame."""

    timestamp: float
    description: str
    audio_transcript: Optional[str] = None
    audio_mode: Optional[str] = None
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        d: dict = {
            "timestamp": round(self.timestamp, 3),
            "description": self.description,
            "latency_ms": round(self.latency_ms, 1),
        }
        if self.audio_transcript is not None:
            d["audio_transcript"] = self.audio_transcript
        if self.audio_mode is not None:
            d["audio_mode"] = self.audio_mode
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FrameRecord":
        return cls(
            timestamp=float(d["timestamp"]),
            description=str(d.get("description", "")),
            audio_transcript=d.get("audio_transcript"),
            audio_mode=d.get("audio_mode"),
            latency_ms=float(d.get("latency_ms", 0.0)),
        )


class AnalysisCache:
    """Read / write analysis results for a single video to a JSON file.

    The cache is intentionally simple — one JSON file, no external DB.
    Complex querying is out of scope; the cache just preserves work across
    sessions.

    Usage::

        cache = AnalysisCache.for_video("myvideo.mp4", "./output/")
        if not cache.is_complete:
            # ... run analysis ...
            cache.add_frame(result)
            cache.mark_complete()
            cache.save()
    """

    def __init__(self, cache_path: Path, thumbnail_path: Path) -> None:
        self.cache_path = cache_path
        self.thumbnail_path = thumbnail_path
        self._frames: list[FrameRecord] = []
        self._full_audio_transcript: Optional[str] = None
        self._summary: Optional[str] = None
        self._metadata: dict = {}
        self._complete: bool = False

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def for_video(cls, video_path: str, output_dir: str) -> "AnalysisCache":
        """Return a cache backed by ``<output_dir>/<stem>_analysis.json``.

        Loads existing cache data if the file already exists so the analysis
        pass can be resumed or skipped entirely on subsequent runs.
        """
        stem = Path(video_path).stem
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        obj = cls(
            cache_path=out / f"{stem}_analysis.json",
            thumbnail_path=out / f"{stem}_thumbnail.jpg",
        )
        if obj.cache_path.exists():
            obj._load()
        return obj

    def _load(self) -> None:
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return  # Treat corrupt or missing cache as empty
        self._metadata = data.get("metadata", {})
        self._frames = [FrameRecord.from_dict(f) for f in data.get("frames", [])]
        self._full_audio_transcript = data.get("full_audio_transcript")
        self._summary = data.get("summary")
        self._complete = bool(data.get("analysis_complete", False))

    def save(self) -> None:
        """Atomically write the cache to disk (via a temp file + rename)."""
        data: dict = {
            "version": _CACHE_VERSION,
            "metadata": self._metadata,
            "frames": [f.to_dict() for f in self._frames],
            "analysis_complete": self._complete,
        }
        if self._full_audio_transcript is not None:
            data["full_audio_transcript"] = self._full_audio_transcript
        if self._summary is not None:
            data["summary"] = self._summary

        # Write to a temp file first to avoid corruption on crash
        tmp = self.cache_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.cache_path)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def set_metadata(
        self,
        *,
        video_path: str,
        model_id: str,
        backend: str,
        audio_enabled: bool,
    ) -> None:
        self._metadata = {
            "video_path": str(video_path),
            "video_name": Path(video_path).name,
            "model_id": model_id,
            "backend": backend,
            "analyzed_at": datetime.now(tz=timezone.utc).isoformat(),
            "audio_enabled": audio_enabled,
        }

    def add_frame(self, result: InferenceResult) -> None:
        """Append one analysed frame to the cache."""
        self._frames.append(
            FrameRecord(
                timestamp=result.timestamp,
                description=result.description,
                audio_transcript=result.audio_transcript,
                audio_mode=result.audio_mode,
                latency_ms=result.latency_ms,
            )
        )

    def set_full_audio_transcript(self, text: str) -> None:
        self._full_audio_transcript = text

    def set_summary(self, text: str) -> None:
        self._summary = text

    def mark_complete(self) -> None:
        """Mark the analysis pass as finished and record the frame count."""
        self._metadata["total_frames_analyzed"] = len(self._frames)
        self._complete = True

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    @property
    def is_complete(self) -> bool:
        """True when the frame-analysis pass has finished successfully."""
        return self._complete

    @property
    def frames(self) -> list[FrameRecord]:
        return list(self._frames)

    @property
    def full_audio_transcript(self) -> Optional[str]:
        return self._full_audio_transcript

    @property
    def summary(self) -> Optional[str]:
        return self._summary

    @property
    def metadata(self) -> dict:
        return dict(self._metadata)
