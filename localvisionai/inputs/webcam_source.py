"""Webcam / USB camera source using OpenCV.

OpenCV's VideoCapture is synchronous. Every read is offloaded to the asyncio
thread-pool so the event loop stays responsive while waiting for the next
camera frame. The generator runs until:
  - the camera is disconnected (read returns False), or
  - the consumer cancels the task (asyncio.CancelledError propagates).
"""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator, Tuple

from PIL import Image

from localvisionai.exceptions import SourceOpenError, SourceReadError
from localvisionai.utils.logging import get_logger
from .base import AbstractVideoSource

logger = get_logger(__name__)


class WebcamSource(AbstractVideoSource):
    """
    Streams frames from a local USB / built-in webcam via OpenCV.

    Args:
        device_index: OpenCV device index (0 = system default webcam).
    """

    def __init__(self, device_index: int = 0) -> None:
        self._device_index = device_index
        self._cap = None
        self._fps: float = 30.0
        self._width: int = 0
        self._height: int = 0

    async def open(self) -> None:
        try:
            import cv2  # type: ignore
        except ImportError:
            raise SourceOpenError(
                "opencv-python is not installed. Install it with: pip install opencv-python"
            )

        loop = asyncio.get_event_loop()
        try:
            cap, fps, w, h = await loop.run_in_executor(None, self._open_sync)
        except Exception as e:
            raise SourceOpenError(
                f"Cannot open webcam at device index {self._device_index}: {e}"
            ) from e

        self._cap = cap
        self._fps = fps
        self._width = w
        self._height = h
        logger.info(
            f"Opened webcam device {self._device_index} — "
            f"{w}x{h} @ {fps:.1f} fps"
        )

    def _open_sync(self):
        import cv2  # type: ignore
        cap = cv2.VideoCapture(self._device_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"OpenCV could not open device index {self._device_index}. "
                "Check that the camera is plugged in and not in use by another app."
            )
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return cap, fps, w, h

    async def close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    @property
    def metadata(self) -> dict:
        return {
            "duration": None,  # Live stream — no known duration
            "fps": self._fps,
            "width": self._width,
            "height": self._height,
            "codec": "webcam",
        }

    async def stream(self) -> AsyncGenerator[Tuple[Image.Image, float], None]:
        if self._cap is None:
            raise SourceReadError("WebcamSource is not open. Call open() first.")

        loop = asyncio.get_event_loop()
        import time
        start_wall = time.perf_counter()

        def _read_frame():
            """Blocking read — must be called in executor."""
            import cv2  # type: ignore
            ok, frame_bgr = self._cap.read()
            if not ok:
                return None, 0.0
            # Convert BGR → RGB then PIL
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            elapsed = time.perf_counter() - start_wall
            return pil_img, elapsed

        while True:
            try:
                img, ts = await loop.run_in_executor(None, _read_frame)
                if img is None:
                    logger.warning("Webcam read returned False — camera disconnected.")
                    break
                yield img, ts
            except asyncio.CancelledError:
                break
            except Exception as e:
                raise SourceReadError(f"Webcam read error: {e}") from e
