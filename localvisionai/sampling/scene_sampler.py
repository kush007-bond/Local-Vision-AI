"""Scene-change sampler — triggers on significant visual transitions.

Compares consecutive frames using a normalised RGB histogram intersection.
When the colour-palette difference between adjacent frames exceeds *threshold*
(expressed as a percentage, 0–100), the frame is considered a scene boundary
and forwarded for inference.

A *min_interval* guard prevents burst submissions in rapid-cut sequences.

Algorithm
---------
1. Resize each frame to a 160×90 thumbnail and compute a 64-bin-per-channel
   RGB histogram (192 values total, each normalised so per-channel bins sum to 1).
2. Compute histogram intersection distance:
       score = (1 − Σ min(h1_i, h2_i) / 3) × 100
   Returns a value in [0, 100] where 0 = identical, 100 = totally different.
3. If score ≥ threshold  AND  (ts − last_sent) ≥ min_interval → process.

No external dependencies beyond Pillow (already a project requirement).
"""

from __future__ import annotations

from typing import Optional

from PIL import Image

from .base import AbstractFrameSampler

# Number of histogram bins per channel (must evenly divide 256)
_BINS = 64
_BIN_WIDTH = 256 // _BINS

# Tiny epsilon to guard against division by zero in chi-squared
_EPS = 1e-6


def _rgb_histogram(frame: Image.Image) -> list[float]:
    """Return a normalised 192-value RGB histogram for *frame*.

    192 = 64 bins × 3 channels (R, G, B).  Each bin value is in [0, 1]
    and all bins for a single channel sum to 1.
    """
    # Resize to a small thumbnail first to speed up calculation
    thumb = frame.resize((160, 90), Image.BILINEAR)
    hist: list[int] = thumb.histogram()  # returns 256*3 values (R then G then B)
    total_pixels = thumb.width * thumb.height

    result: list[float] = []
    for channel in range(3):            # R=0, G=1, B=2
        offset = channel * 256
        # Bin 256 fine buckets into _BINS coarser buckets
        for b in range(_BINS):
            lo = offset + b * _BIN_WIDTH
            hi = lo + _BIN_WIDTH
            bucket_sum = sum(hist[lo:hi])
            result.append(bucket_sum / (total_pixels + _EPS))

    return result


def _histogram_diff(h1: list[float], h2: list[float]) -> float:
    """Scene-change score in the range [0, 100].

    Uses histogram *intersection* distance:
        score = (1 − Σ min(h1_i, h2_i) / n_channels) × 100

    A score of 0 means identical histograms; 100 means completely different.
    With 3 channels each summing to 1, the unscaled intersection is in [0, 3].
    Dividing by 3 normalises to [0, 1] before the ×100 scaling, so the
    returned value maps cleanly to the ``scene_threshold`` config value where
    ``30.0`` ≈ 30 % colour-palette change — a comfortable default for detecting
    hard cuts without over-triggering on minor motion.
    """
    n_channels = 3
    intersection = sum(min(a, b) for a, b in zip(h1, h2))
    # intersection is in [0, n_channels] (one unit per channel when histos match)
    return (1.0 - intersection / (n_channels + _EPS)) * 100.0



class SceneSampler(AbstractFrameSampler):
    """Forward frames only when a scene change is detected.

    Args:
        threshold:    Chi-squared distance above which a scene change is
                      declared.  Typical useful range is 5 – 60.
                      Lower values → more sensitive (more frames sent).
                      Default 30.0 matches the config default.
        min_interval: Minimum seconds between two accepted frames.
                      Guards against rapid-cut bursts.  Default 0.5 s.
    """

    def __init__(
        self,
        threshold: float = 30.0,
        min_interval: float = 0.5,
    ) -> None:
        if threshold <= 0:
            raise ValueError(f"threshold must be positive, got {threshold}")
        if min_interval < 0:
            raise ValueError(f"min_interval must be non-negative, got {min_interval}")

        self.threshold = threshold
        self.min_interval = min_interval

        self._prev_hist: Optional[list[float]] = None
        self._last_sent: float = -min_interval  # Ensures the very first frame passes

    # ------------------------------------------------------------------
    # AbstractFrameSampler interface
    # ------------------------------------------------------------------

    def should_process(self, frame: Image.Image, timestamp: float) -> bool:
        """Return True when a scene-change is detected (or on the first frame)."""
        hist = _rgb_histogram(frame)

        # First frame always passes
        if self._prev_hist is None:
            self._prev_hist = hist
            self._last_sent = timestamp
            return True

        # Minimum interval guard
        if (timestamp - self._last_sent) < self.min_interval:
            self._prev_hist = hist
            return False

        # Scene-change detection
        dist = _histogram_diff(self._prev_hist, hist)
        self._prev_hist = hist

        if dist >= self.threshold:
            self._last_sent = timestamp
            return True

        return False

    def reset(self) -> None:
        """Reset sampler state for a new video."""
        self._prev_hist = None
        self._last_sent = -self.min_interval
