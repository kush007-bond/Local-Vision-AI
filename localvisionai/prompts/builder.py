"""Prompt construction logic — builds the final prompt sent to the model."""

from __future__ import annotations

from localvisionai.config import PromptConfig
from .templates import get_system_prompt, format_verbosity_prompt


def build_prompt(
    config: PromptConfig,
    context_summary: str | None = None,
) -> tuple[str, str | None]:
    """
    Build the (user_prompt, system_prompt) tuple for a given frame.

    Args:
        config: The PromptConfig from PipelineConfig.
        context_summary: Rolling context from prior frames (sliding window mode).

    Returns:
        (user_prompt, system_prompt) — system_prompt may be None.
    """
    user_prompt = config.user

    # Inject sliding window context into user prompt
    if context_summary and config.context_mode == "sliding_window":
        user_prompt = (
            f"Previous context: {context_summary}\n\n"
            f"Current frame: {user_prompt}"
        )

    system_prompt = get_system_prompt(
        model_id="",  # Generic — will be overridden by adapter if needed
        override=config.system,
    )

    return user_prompt, system_prompt
