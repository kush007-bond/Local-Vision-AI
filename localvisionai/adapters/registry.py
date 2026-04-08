"""Adapter registry — maps backend name strings to adapter classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .ollama_adapter import OllamaAdapter
from .transformers_adapter import HuggingFaceAdapter

# These are imported lazily to avoid requiring all optional packages
_LAZY_ADAPTERS = {
    "llamacpp": ("localvisionai.adapters.llamacpp_adapter", "LlamaCppAdapter"),
    "mlx": ("localvisionai.adapters.mlx_adapter", "MLXAdapter"),
    "vllm": ("localvisionai.adapters.vllm_adapter", "VLLMAdapter"),
    "openai": ("localvisionai.adapters.openai_adapter", "OpenAIAdapter"),
    "anthropic": ("localvisionai.adapters.anthropic_adapter", "AnthropicAdapter"),
    "gemini": ("localvisionai.adapters.gemini_adapter", "GeminiAdapter"),
    "lmstudio": ("localvisionai.adapters.lmstudio_adapter", "LMStudioAdapter"),
}

# Eagerly available adapters
REGISTRY: dict = {
    "ollama": OllamaAdapter,
    "transformers": HuggingFaceAdapter,
}


def get_adapter(backend: str, **kwargs) -> object:
    """
    Instantiate an adapter by backend name.

    Args:
        backend: One of 'ollama', 'transformers', 'llamacpp', 'mlx', 'vllm'.
        **kwargs: Keyword arguments forwarded to the adapter's __init__.

    Returns:
        An AbstractModelAdapter instance.

    Raises:
        ValueError: If the backend name is unknown.
        ImportError: If the backend's optional dependency is not installed.
    """
    if backend in REGISTRY:
        return REGISTRY[backend](**kwargs)

    if backend in _LAZY_ADAPTERS:
        module_path, class_name = _LAZY_ADAPTERS[backend]
        import importlib
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        REGISTRY[backend] = cls  # Cache for next call
        return cls(**kwargs)

    available = list(REGISTRY.keys()) + list(_LAZY_ADAPTERS.keys())
    raise ValueError(
        f"Unknown backend '{backend}'. Available backends: {available}"
    )
