"""JSON timeline output handler — writes structured results to disk."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Optional

from localvisionai.adapters.base import InferenceResult
from localvisionai.utils.logging import get_logger
from .base import AbstractOutputHandler

logger = get_logger(__name__)


class JSONOutput(AbstractOutputHandler):
    """
    Writes inference results to a JSON file.

    For long jobs (>60s), flushes incrementally every `flush_every` seconds
    using an atomic temp-file + rename pattern to prevent data loss on crash.

    Output format:
    {
        "job_id": "j_abc123",
        "model": "gemma3",
        "backend": "ollama",
        "generated_at": "2026-04-07T12:05:30Z",
        "frames": [
            {"timestamp": 0.0, "description": "...", "latency_ms": 1240, "token_count": 11},
            ...
        ]
    }
    """

    def __init__(
        self,
        output_dir: str,
        flush_every: int = 30,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._flush_every = flush_every
        self._job_id: Optional[str] = None
        self._output_path: Optional[Path] = None
        self._buffer: list[dict] = []
        self._metadata: dict = {}
        self._last_flush: float = 0.0

    async def open(self, job_id: str) -> None:
        self._job_id = job_id
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._output_path = self._output_dir / f"{job_id}.json"
        self._buffer = []
        self._last_flush = time.time()
        logger.info(f"JSON output: {self._output_path}")

    def set_metadata(self, model_id: str, backend: str, source: str) -> None:
        """Call after open() to set job-level metadata."""
        import datetime
        self._metadata = {
            "job_id": self._job_id,
            "source": source,
            "model": model_id,
            "backend": backend,
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    async def handle(self, result: InferenceResult) -> None:
        self._buffer.append(result.to_dict())

        # Incremental flush for long jobs
        if time.time() - self._last_flush >= self._flush_every:
            await self._flush_to_disk()
            self._last_flush = time.time()

    async def close(self) -> None:
        """Final flush on job completion."""
        await self._flush_to_disk()
        logger.info(f"JSON output written: {self._output_path} ({len(self._buffer)} frames total)")

    async def _flush_to_disk(self) -> None:
        if self._output_path is None:
            return

        payload = {**self._metadata, "frames": self._buffer}
        tmp_path = self._output_path.with_suffix(".tmp")

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._write_atomic, tmp_path, payload)

    def _write_atomic(self, tmp_path: Path, payload: dict) -> None:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        # Atomic rename
        tmp_path.replace(self._output_path)
