"""
LocalVisionAI — Local AI-powered video understanding pipeline.

Quick usage:

    from localvisionai import Pipeline, PipelineConfig

    config = PipelineConfig.from_yaml("configs/default.yaml")
    import asyncio
    asyncio.run(Pipeline(config).run())
"""

from localvisionai.pipeline import Pipeline
from localvisionai.config import PipelineConfig
from localvisionai.exceptions import (
    LocalVisionAIError,
    ModelNotFoundError,
    ModelInferenceError,
    SourceOpenError,
    SourceReadError,
    ConfigValidationError,
)

__version__ = "0.1.0"
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
