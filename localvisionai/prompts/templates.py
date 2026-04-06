"""Per-model chat template formatting."""

from __future__ import annotations

# Default templates for model families.
# These handle the raw text wrapping before the processor's apply_chat_template is used.
# For Ollama and most HuggingFace models, the SDK/processor handles this automatically.
# This module provides manual template overrides for edge cases.

DEFAULT_SYSTEM_PROMPT = "You are a concise video analyst. Describe what is happening in the image."

# Model family → system prompt hint mapping
MODEL_SYSTEM_HINTS: dict[str, str] = {
    "gemma": "You are a helpful visual assistant. Describe the image briefly and accurately.",
    "llava": "A chat between a curious user and an artificial intelligence assistant that can understand images.",
    "qwen": "You are a helpful video description assistant. Be concise and factual.",
    "internvl": "You are an AI assistant specialized in visual understanding.",
    "minicpm": "You are a helpful multimodal assistant.",
    "moondream": "You are a visual description assistant.",
}


def get_system_prompt(model_id: str, override: str | None = None) -> str | None:
    """
    Return appropriate system prompt for a model.

    Args:
        model_id: The model identifier string (e.g. 'gemma3', 'llava:13b').
        override: If provided, always returned directly.

    Returns:
        System prompt string, or None to use no system prompt.
    """
    if override is not None:
        return override

    model_lower = model_id.lower()
    for family, hint in MODEL_SYSTEM_HINTS.items():
        if family in model_lower:
            return hint

    return DEFAULT_SYSTEM_PROMPT


def format_verbosity_prompt(base_prompt: str, verbosity: str = "normal") -> str:
    """
    Append verbosity instructions to a prompt.

    Args:
        base_prompt: The core prompt from the user.
        verbosity: 'terse' (1 word), 'normal' (1 sentence), 'detailed' (paragraph).
    """
    suffixes = {
        "terse": " Reply with at most 3 words.",
        "normal": " Reply in one sentence.",
        "detailed": " Provide a detailed description in 2-4 sentences.",
    }
    suffix = suffixes.get(verbosity, "")
    if suffix and not base_prompt.endswith(suffix.strip()):
        return base_prompt.rstrip(".") + suffix
    return base_prompt
