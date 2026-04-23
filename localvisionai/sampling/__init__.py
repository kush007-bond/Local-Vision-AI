"""Sampling layer package."""

from localvisionai.config import SamplingConfig
from .base import AbstractFrameSampler
from .uniform_sampler import UniformSampler
from .keyframe_sampler import KeyframeSampler
from .scene_sampler import SceneSampler
from .adaptive_sampler import AdaptiveSampler


def build_sampler(config: SamplingConfig) -> AbstractFrameSampler:
    """Factory: construct the correct AbstractFrameSampler from config."""
    if config.strategy == "uniform":
        return UniformSampler(fps=config.fps)
    elif config.strategy == "keyframe":
        return KeyframeSampler(min_interval=config.min_interval)
    elif config.strategy == "scene":
        return SceneSampler(
            threshold=config.scene_threshold,
            min_interval=config.min_interval,
        )
    elif config.strategy == "adaptive":
        return AdaptiveSampler(target_fps=config.fps)
    else:
        raise ValueError(f"Unknown sampling strategy: {config.strategy!r}")


__all__ = [
    "AbstractFrameSampler",
    "UniformSampler",
    "KeyframeSampler",
    "SceneSampler",
    "AdaptiveSampler",
    "build_sampler",
]
