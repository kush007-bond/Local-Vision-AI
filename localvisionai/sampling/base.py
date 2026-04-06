"""Abstract base class for frame samplers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image


class AbstractFrameSampler(ABC):
    """
    Determines which frames from a video source should be sent to the model.

    Called once per decoded frame. The sampler maintains internal state
    (e.g. last sent timestamp, previous frame data) to make decisions.
    """

    @abstractmethod
    def should_process(self, frame: Image.Image, timestamp: float) -> bool:
        """
        Return True if this frame should be queued for model inference.

        Args:
            frame: The decoded frame as a PIL Image.
            timestamp: Seconds elapsed from the start of the source.
        """
        ...

    def reset(self) -> None:
        """
        Reset internal state. Called when starting a new video.
        Override if your sampler maintains stateful history.
        """
        pass
