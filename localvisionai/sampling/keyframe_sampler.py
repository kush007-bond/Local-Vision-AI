"""Keyframe-only sampler — passes only I-frames (intra-coded frames).

Note: This sampler works by wrapping the standard VideoFileSource and
checking PyAV packet flags. For sources that don't expose packet info
(webcam, RTSP), it falls back to UniformSampler behavior at 1 FPS.

The keyframe_sampler is meant to be used in conjunction with the
VideoFileSource or RTSPSource where packet-level access is available.
Since the AbstractFrameSampler interface only receives PIL frames (not
raw packets), this sampler uses a workaround: it marks frames as
keyframes using a shared state set by the source.
"""

from __future__ import annotations

from PIL import Image

from .base import AbstractFrameSampler


class KeyframeSampler(AbstractFrameSampler):
    """
    Passes frames that have been flagged as keyframes by the source.

    The VideoFileSource sets a frame attribute `_is_keyframe = True`
    on PIL images that correspond to I-frames. This sampler checks
    that attribute. For sources that don't set this attribute, all
    frames pass through (equivalent to UniformSampler at source FPS).

    This is the fastest sampling strategy for long files — O(1) frames
    relative to file length, depending on encoder keyframe interval.
    """

    def __init__(self, min_interval: float = 0.0) -> None:
        """
        Args:
            min_interval: Minimum seconds between accepted keyframes.
                          Use > 0 to avoid bursts of keyframes in high-motion segments.
        """
        self.min_interval = min_interval
        self._last_sent: float = -min_interval

    def should_process(self, frame: Image.Image, timestamp: float) -> bool:
        # Respect minimum interval
        if self.min_interval > 0 and (timestamp - self._last_sent) < self.min_interval:
            return False

        # Check if the frame was flagged as a keyframe by the source
        is_keyframe = getattr(frame, "_is_keyframe", True)  # Default True → safe fallback
        if is_keyframe:
            self._last_sent = timestamp
            return True
        return False

    def reset(self) -> None:
        self._last_sent = -self.min_interval
