"""
Audio analysis package for LocalVisionAI.

Provides time-aligned audio extraction from video sources and lazy segmentation
into per-frame chunks. Audio chunks are forwarded natively to multimodal model
adapters (e.g. Gemini, GPT-4o, Claude) that declare `supports_audio = True`.
"""

from .base import AudioChunk, AbstractAudioExtractor
from .ffmpeg_extractor import FfmpegAudioExtractor
from .segmenter import AudioSegmenter

__all__ = [
    "AudioChunk",
    "AbstractAudioExtractor",
    "FfmpegAudioExtractor",
    "AudioSegmenter",
]
