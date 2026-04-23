"""Adaptive FPS sampler — dynamically adjusts the sampling rate based on
actual model inference latency.

Motivation
----------
On CPU-only machines, vision models are often slower than the configured FPS
target.  A static UniformSampler would queue frames faster than the consumer
can process them, causing queue growth, memory pressure, and falling behind
real-time.

AdaptiveSampler keeps the pipeline *sustainable*: it starts at the configured
``target_fps`` and continuously measures real inference latency via
``record_latency()``.  When the model proves unable to keep up it lowers the
effective FPS; when headroom is available it cautiously raises it back up.

Control law
-----------
After every completed inference the ``record_latency`` method is called with
the measured wall-clock seconds.  The *effective interval* between accepted
frames is adjusted as follows:

    new_interval = α × current_interval + (1 − α) × measured_latency

where α = ``smoothing`` (default 0.7) is an exponential moving-average weight.

The interval is clamped to [1 / max_fps, 1 / min_fps] to prevent runaway
behaviour.

Usage in pipeline.py
--------------------
``Pipeline._maybe_adjust_fps`` calls ``sampler.record_latency(elapsed_s)`` if
the sampler supports it.  No change is needed for samplers that don't.
"""

from __future__ import annotations

from PIL import Image

from .base import AbstractFrameSampler

# Hard limits on the effective FPS
_MIN_FPS = 0.1   # Never slower than once per 10 seconds
_MAX_FPS = 30.0  # Never faster than 30 fps


class AdaptiveSampler(AbstractFrameSampler):
    """Uniform sampler with latency-driven FPS adaptation.

    Args:
        target_fps: Initial (and maximum) frames-per-second rate.
        min_fps:    Floor on the effective FPS (prevents stalling).
        smoothing:  EMA smoothing factor (0 < α < 1).  Higher values
                    make adaptation slower but more stable.
    """

    def __init__(
        self,
        target_fps: float = 1.0,
        min_fps: float = _MIN_FPS,
        smoothing: float = 0.7,
    ) -> None:
        if target_fps <= 0:
            raise ValueError(f"target_fps must be positive, got {target_fps}")
        if min_fps <= 0 or min_fps > target_fps:
            min_fps = min(target_fps, _MIN_FPS)

        self.target_fps = target_fps
        self.min_fps = max(min_fps, _MIN_FPS)
        self.max_fps = min(target_fps, _MAX_FPS)
        self.smoothing = max(0.0, min(1.0, smoothing))

        # Effective sampling interval (seconds between accepted frames)
        self._interval: float = 1.0 / target_fps
        self._last_sent: float = -self._interval  # Ensures first frame is sent

        # Statistics (exposed for logging / tests)
        self.frames_accepted: int = 0
        self.frames_rejected: int = 0
        self._latency_samples: int = 0

    # ------------------------------------------------------------------
    # AbstractFrameSampler interface
    # ------------------------------------------------------------------

    def should_process(self, frame: Image.Image, timestamp: float) -> bool:
        if (timestamp - self._last_sent) >= self._interval:
            self._last_sent = timestamp
            self.frames_accepted += 1
            return True
        self.frames_rejected += 1
        return False

    def reset(self) -> None:
        self._interval = 1.0 / self.target_fps
        self._last_sent = -self._interval
        self.frames_accepted = 0
        self.frames_rejected = 0
        self._latency_samples = 0

    # ------------------------------------------------------------------
    # Adaptive control — called by Pipeline after each inference
    # ------------------------------------------------------------------

    def record_latency(self, elapsed_s: float) -> None:
        """Update the effective interval based on measured inference time.

        Args:
            elapsed_s: Wall-clock seconds taken by the last inference call.
        """
        if elapsed_s <= 0:
            return

        self._latency_samples += 1

        # EMA update
        self._interval = (
            self.smoothing * self._interval
            + (1.0 - self.smoothing) * elapsed_s
        )

        # Clamp to [min_fps, max_fps]
        min_interval = 1.0 / self.max_fps
        max_interval = 1.0 / self.min_fps
        self._interval = max(min_interval, min(max_interval, self._interval))

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def effective_fps(self) -> float:
        """Current effective FPS (may be lower than target_fps)."""
        return 1.0 / self._interval if self._interval > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"AdaptiveSampler("
            f"target_fps={self.target_fps}, "
            f"effective_fps={self.effective_fps:.2f}, "
            f"interval={self._interval:.3f}s, "
            f"latency_samples={self._latency_samples})"
        )
