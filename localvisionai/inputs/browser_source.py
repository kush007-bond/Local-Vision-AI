"""Browser-captured webcam source.

On Windows (and some Linux configurations), the browser's getUserMedia and
OpenCV's VideoCapture cannot share the same camera simultaneously. This source
avoids that conflict entirely: the browser captures frames from its own live
stream and pushes them to the backend via HTTP, so OpenCV is never used for
webcam capture.
"""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator, Tuple

from PIL import Image

from .base import AbstractVideoSource


class BrowserCaptureSource(AbstractVideoSource):
    """
    Receives PIL frames pushed by the browser via the /api/jobs/{id}/frame endpoint.

    The pipeline treats this exactly like any other source — stream() is an
    async generator that yields (PIL.Image, float) tuples as frames arrive.
    The stream ends when stop() is called (job cancelled/completed) or when
    no frame arrives within IDLE_TIMEOUT_S seconds (client disconnected).
    """

    IDLE_TIMEOUT_S = 60.0  # End stream if no frame arrives for this long

    def __init__(self) -> None:
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=32)
        self._stopped = False

    async def open(self) -> None:
        pass  # No hardware resource to open

    async def close(self) -> None:
        pass

    def stop(self) -> None:
        """Signal the stream to end (e.g. when the job is cancelled)."""
        self._stopped = True
        try:
            self._queue.put_nowait(None)  # Wake up the stream() coroutine
        except asyncio.QueueFull:
            pass

    async def push_frame(self, image: Image.Image, timestamp: float) -> None:
        """Push a browser-captured frame into the pipeline queue. Drops if full."""
        if self._stopped:
            return
        try:
            self._queue.put_nowait((image, timestamp))
        except asyncio.QueueFull:
            pass  # Drop newest frame; model is slower than capture rate

    @property
    def metadata(self) -> dict:
        return {
            "duration": None,
            "fps": None,
            "width": None,
            "height": None,
            "codec": "browser-capture",
        }

    async def stream(self) -> AsyncGenerator[Tuple[Image.Image, float], None]:
        while True:
            try:
                item = await asyncio.wait_for(
                    self._queue.get(), timeout=self.IDLE_TIMEOUT_S
                )
            except asyncio.TimeoutError:
                break  # Client stopped sending frames
            except asyncio.CancelledError:
                break

            if item is None:  # Stop sentinel from stop()
                break

            yield item
