"""Abstract base class for all video input sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional, Tuple

from PIL import Image


class AbstractVideoSource(ABC):
    """
    Base class for all video input sources.

    Implementations must be async context managers and async generators.
    Every source yields (PIL.Image, timestamp_seconds) tuples.
    """

    @abstractmethod
    async def open(self) -> None:
        """Initialize any resources (file handles, connections, capture devices)."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release all resources cleanly. Must be idempotent."""
        ...

    @abstractmethod
    async def stream(self) -> AsyncGenerator[Tuple[Image.Image, float], None]:
        """
        Async generator that yields (frame, timestamp) tuples.

        - frame: PIL.Image in RGB mode
        - timestamp: seconds elapsed from the start of the source

        The generator should run until the source is exhausted or closed.
        For live sources (webcam, RTSP), it runs indefinitely until close() is called.
        """
        ...

    @property
    @abstractmethod
    def metadata(self) -> dict:
        """
        Return source metadata. Keys where available:
            duration (float | None), fps (float | None),
            width (int | None), height (int | None), codec (str | None)
        """
        ...

    # ------------------------------------------------------------------
    # Async context manager support
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "AbstractVideoSource":
        await self.open()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
