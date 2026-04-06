"""Video file source implementation using PyAV (FFmpeg bindings)."""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator, Optional, Tuple

from PIL import Image

from localvisionai.exceptions import SourceOpenError, SourceReadError
from localvisionai.utils.logging import get_logger
from .base import AbstractVideoSource

logger = get_logger(__name__)


class VideoFileSource(AbstractVideoSource):
    """
    Reads any video file format supported by FFmpeg.
    Uses PyAV for fast frame decoding. All blocking I/O runs in a thread pool
    to avoid blocking the asyncio event loop.

    Supports: MP4, AVI, MKV, MOV, WebM, MTS, and more.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._container = None
        self._metadata: Optional[dict] = None

    async def open(self) -> None:
        try:
            import av
        except ImportError:
            raise SourceOpenError(
                "PyAV is not installed. Install it with: pip install av"
            )
        loop = asyncio.get_event_loop()
        try:
            self._container = await loop.run_in_executor(None, self._open_sync)
        except Exception as e:
            raise SourceOpenError(f"Cannot open video file '{self.path}': {e}") from e

        # Cache metadata immediately after open
        self._metadata = self._read_metadata()
        logger.info(
            f"Opened video file: {self.path}",
            extra={
                "duration": self._metadata.get("duration"),
                "fps": self._metadata.get("fps"),
                "resolution": f"{self._metadata.get('width')}x{self._metadata.get('height')}",
            },
        )

    def _open_sync(self):
        import av
        return av.open(self.path)

    async def close(self) -> None:
        if self._container:
            try:
                self._container.close()
            except Exception:
                pass
            self._container = None

    def _read_metadata(self) -> dict:
        try:
            video_stream = self._container.streams.video[0]
            duration = None
            if self._container.duration:
                duration = float(self._container.duration / 1_000_000)  # microseconds → seconds
            fps = None
            if video_stream.average_rate:
                fps = float(video_stream.average_rate)
            return {
                "duration": duration,
                "fps": fps,
                "width": video_stream.width,
                "height": video_stream.height,
                "codec": video_stream.codec_context.name if video_stream.codec_context else None,
            }
        except (IndexError, AttributeError):
            return {"duration": None, "fps": None, "width": None, "height": None, "codec": None}

    @property
    def metadata(self) -> dict:
        if self._metadata is None:
            return {"duration": None, "fps": None, "width": None, "height": None, "codec": None}
        return self._metadata

    async def stream(self) -> AsyncGenerator[Tuple[Image.Image, float], None]:
        if self._container is None:
            raise SourceReadError("Source is not open. Call open() first.")

        loop = asyncio.get_event_loop()

        try:
            video_stream = self._container.streams.video[0]
        except IndexError:
            raise SourceReadError(f"No video stream found in '{self.path}'")

        def _decode_generator():
            for packet in self._container.demux(video_stream):
                try:
                    for frame in packet.decode():
                        if frame.pts is None:
                            continue
                        ts = float(frame.pts * video_stream.time_base)
                        # Convert to PIL — to_image() returns RGB
                        img = frame.to_image()
                        yield img, ts
                except Exception as e:
                    logger.warning(f"Failed to decode frame packet: {e}")
                    continue

        gen = _decode_generator()

        while True:
            try:
                frame, ts = await loop.run_in_executor(None, next, gen)
                yield frame, ts
            except StopIteration:
                logger.info(f"Video file exhausted: {self.path}")
                break
            except Exception as e:
                raise SourceReadError(f"Error reading frame from '{self.path}': {e}") from e
