"""
Audio analysis package for LocalVisionAI.

Provides time-aligned audio extraction from video sources, lazy segmentation
into per-frame chunks, and transcription via faster-whisper (or openai-whisper).

Audio runs alongside, not instead of, video frames. Two modes are supported:
    - native:     raw audio bytes forwarded to multimodal models (OpenAI/Gemini)
    - transcribe: audio transcribed to text and injected into the prompt

See AUDIO_FEATURE_ARCHITECTURE.md for the full design.
"""

from .base import AudioChunk, AbstractAudioExtractor
from .ffmpeg_extractor import FfmpegAudioExtractor
from .segmenter import AudioSegmenter
from .transcriber import WhisperTranscriber

__all__ = [
    "AudioChunk",
    "AbstractAudioExtractor",
    "FfmpegAudioExtractor",
    "AudioSegmenter",
    "WhisperTranscriber",
]
