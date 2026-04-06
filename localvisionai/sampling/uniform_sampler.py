"""Uniform FPS frame sampler — extracts frames at a fixed rate."""

from __future__ import annotations

from PIL import Image

from .base import AbstractFrameSampler


class UniformSampler(AbstractFrameSampler):
    """
    Passes frames to the model at a fixed frames-per-second rate.

    This is the default and simplest strategy. It tracks the timestamp
    of the last processed frame and only approves new frames when
    enough time has elapsed.

    Example:
        fps=1.0  → one frame per second
        fps=0.5  → one frame every two seconds
        fps=2.0  → two frames per second
    """

    def __init__(self, fps: float = 1.0) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        self.fps = fps
        self.interval = 1.0 / fps
        self._last_sent: float = -self.interval  # Ensures the first frame is always sent

    def should_process(self, frame: Image.Image, timestamp: float) -> bool:
        if timestamp - self._last_sent >= self.interval:
            self._last_sent = timestamp
            return True
        return False

    def reset(self) -> None:
        self._last_sent = -self.interval
