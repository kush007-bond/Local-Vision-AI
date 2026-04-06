"""Latency profiling utilities for LocalVisionAI."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, Optional


@dataclass
class LatencyStats:
    """Rolling latency statistics for a pipeline component."""
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0

    def record(self, ms: float) -> None:
        self.count += 1
        self.total_ms += ms
        self.min_ms = min(self.min_ms, ms)
        self.max_ms = max(self.max_ms, ms)

    @property
    def mean_ms(self) -> float:
        return self.total_ms / self.count if self.count else 0.0

    def __str__(self) -> str:
        if not self.count:
            return "no samples"
        return (
            f"n={self.count} mean={self.mean_ms:.0f}ms "
            f"min={self.min_ms:.0f}ms max={self.max_ms:.0f}ms"
        )


class LatencyTracker:
    """
    Context manager that measures elapsed time in milliseconds.

    Usage:
        tracker = LatencyTracker()
        with tracker.measure() as t:
            ... do work ...
        print(f"Took {t.elapsed_ms:.1f}ms")
        print(tracker.stats)
    """

    def __init__(self) -> None:
        self.stats = LatencyStats()
        self.last_ms: float = 0.0

    @contextmanager
    def measure(self) -> Generator["LatencyTracker", None, None]:
        start = time.perf_counter()
        try:
            yield self
        finally:
            self.last_ms = (time.perf_counter() - start) * 1000
            self.stats.record(self.last_ms)

    @property
    def elapsed_ms(self) -> float:
        return self.last_ms


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def format_srt_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
