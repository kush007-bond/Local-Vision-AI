"""Input sources package."""

from localvisionai.config import SourceConfig
from .base import AbstractVideoSource
from .file_source import VideoFileSource


def build_source(config: SourceConfig) -> AbstractVideoSource:
    """Factory: construct the correct AbstractVideoSource from config."""
    if config.type == "file":
        return VideoFileSource(path=config.path)
    elif config.type == "webcam":
        from .webcam_source import WebcamSource
        return WebcamSource(device_index=config.device_index)
    elif config.type == "rtsp":
        from .rtsp_source import RTSPSource
        return RTSPSource(url=config.rtsp_url)
    elif config.type == "url":
        from .url_source import URLSource
        return URLSource(url=config.path)
    elif config.type == "screen":
        from .screen_source import ScreenCaptureSource
        return ScreenCaptureSource()
    elif config.type == "audio":
        from .audio_source import AudioOnlySource
        # Use audio.window_seconds from the root config if available, else default 3.0
        # We pass a default here; pipeline.py will also force audio.enabled=True
        return AudioOnlySource(path=config.path, window_seconds=3.0)
    else:
        raise ValueError(f"Unknown source type: {config.type!r}")


__all__ = ["AbstractVideoSource", "VideoFileSource", "build_source"]
