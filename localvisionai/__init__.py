"""
LocalVisionAI — Local AI-powered video understanding pipeline.

Quick usage:

    from localvisionai import Pipeline, PipelineConfig

    config = PipelineConfig.from_yaml("configs/default.yaml")
    import asyncio
    asyncio.run(Pipeline(config).run())
"""

from localvisionai.exceptions import (
    LocalVisionAIError,
    ModelNotFoundError,
    ModelInferenceError,
    SourceOpenError,
    SourceReadError,
    ConfigValidationError,
)

__version__ = "0.1.0"


def __getattr__(name: str):
    """Lazy imports for Pipeline and PipelineConfig to avoid circular imports at test time."""
    if name == "Pipeline":
        from localvisionai.pipeline import Pipeline
        return Pipeline
    if name == "PipelineConfig":
        from localvisionai.config import PipelineConfig
        return PipelineConfig
    raise AttributeError(f"module 'localvisionai' has no attribute {name!r}")


__all__ = [
    "Pipeline",
    "PipelineConfig",
    "LocalVisionAIError",
    "ModelNotFoundError",
    "ModelInferenceError",
    "SourceOpenError",
    "SourceReadError",
    "ConfigValidationError",
]
