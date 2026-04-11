"""Lazy per-frame audio segmentation over a pre-loaded sample buffer."""

from __future__ import annotations

import io
import wave

import numpy as np

from .base import AudioChunk


class AudioSegmenter:
    """Slices a pre-loaded audio buffer into per-frame AudioChunks.

    The buffer is the interleaved float32 output of FfmpegAudioExtractor.
    Slicing is an O(1) array view — there is no per-frame I/O cost.
    """

    def __init__(self, samples: np.ndarray, sample_rate: int, channels: int) -> None:
        self._samples = samples
        self._sample_rate = int(sample_rate)
        self._channels = max(1, int(channels))
        # Number of frames (sample groups) in the buffer
        self._n_frames = len(samples) // self._channels if len(samples) else 0

    @property
    def duration(self) -> float:
        if self._sample_rate <= 0:
            return 0.0
        return self._n_frames / float(self._sample_rate)

    def get_chunk(self, frame_ts: float, window_seconds: float) -> AudioChunk:
        """Return an AudioChunk ending at `frame_ts`.

        Window: ``[frame_ts - window_seconds, frame_ts]``, clamped to
        ``[0, duration]``. When the buffer is empty, an empty chunk is
        returned so callers can still branch on `transcript`/`is_empty`
        without special-casing.
        """
        if self._n_frames == 0 or window_seconds <= 0:
            return AudioChunk(
                data=b"",
                sample_rate=self._sample_rate,
                channels=self._channels,
                start_ts=max(0.0, frame_ts - window_seconds),
                end_ts=max(0.0, frame_ts),
            )

        end_ts = max(0.0, min(frame_ts, self.duration))
        start_ts = max(0.0, end_ts - window_seconds)

        start_frame = int(start_ts * self._sample_rate)
        end_frame = int(end_ts * self._sample_rate)
        start_sample = start_frame * self._channels
        end_sample = end_frame * self._channels

        slice_view = self._samples[start_sample:end_sample]
        return AudioChunk(
            data=slice_view.tobytes(),
            sample_rate=self._sample_rate,
            channels=self._channels,
            start_ts=start_ts,
            end_ts=end_ts,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def chunk_to_wav_bytes(chunk: AudioChunk) -> bytes:
        """Convert a float32 AudioChunk to a WAV (PCM16) byte string.

        Adapters that need a standard container (OpenAI/Gemini) call this
        at send time. Returns an empty bytes object for empty chunks.
        """
        if chunk.is_empty:
            return b""

        floats = np.frombuffer(chunk.data, dtype=np.float32)
        # Clip and convert to PCM16 — the standard WAV format
        pcm16 = np.clip(floats, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype(np.int16)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(chunk.channels)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(chunk.sample_rate)
            wav.writeframes(pcm16.tobytes())
        return buffer.getvalue()

    @staticmethod
    def chunk_to_numpy(chunk: AudioChunk) -> np.ndarray:
        """Return the chunk samples as a float32 numpy array.

        If the source is multichannel, the result is downmixed to mono by
        averaging — this is what Whisper expects.
        """
        if chunk.is_empty:
            return np.zeros(0, dtype=np.float32)
        floats = np.frombuffer(chunk.data, dtype=np.float32)
        if chunk.channels > 1:
            floats = floats.reshape(-1, chunk.channels).mean(axis=1)
        return floats.astype(np.float32, copy=False)
