"""Adapters package."""

from .base import AbstractModelAdapter, InferenceResult
from .ollama_adapter import OllamaAdapter
from .transformers_adapter import HuggingFaceAdapter
from .registry import get_adapter, REGISTRY

__all__ = [
    "AbstractModelAdapter",
    "InferenceResult",
    "OllamaAdapter",
    "HuggingFaceAdapter",
    "get_adapter",
    "REGISTRY",
]
