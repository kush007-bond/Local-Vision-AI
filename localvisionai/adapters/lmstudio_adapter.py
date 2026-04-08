"""LM Studio backend adapter — OpenAI-compatible local API.

LM Studio exposes an OpenAI-compatible REST API at http://localhost:1234/v1.
This adapter is a thin wrapper around OpenAIAdapter with LM Studio defaults.

Requires: pip install localvisionai[openai]  (reuses the openai package)
No API key needed — LM Studio accepts any non-empty string.

Usage:
    # In LM Studio: load a vision model and start the local server
    localvisionai run --backend lmstudio --model "your-loaded-model-name" --video ...

Notes:
    - The model_id must match the model name shown in LM Studio's server tab.
    - Default endpoint: http://localhost:1234/v1  (change with --api-base).
    - Vision support depends on the loaded model; use LLaVA, Qwen2-VL,
      InternVL2 or any other multimodal GGUF / MLX model in LM Studio.
"""

from __future__ import annotations

from typing import Optional

from .openai_adapter import OpenAIAdapter

_DEFAULT_LMS_BASE = "http://localhost:1234/v1"
_LMS_DUMMY_KEY = "lm-studio"


class LMStudioAdapter(OpenAIAdapter):
    """
    Vision adapter for LM Studio's local OpenAI-compatible server.

    Inherits all streaming and multi-frame logic from OpenAIAdapter.
    Only the defaults differ: base_url points to localhost:1234 and no
    real API key is required.

    Usage:
        adapter = LMStudioAdapter("llava-v1.6-mistral-7b")
        await adapter.load()
        async for token in adapter.infer(frame, "Describe this frame."):
            print(token, end="", flush=True)
    """

    def __init__(
        self,
        model_id: str = "local-model",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_new_tokens: int = 512,
    ) -> None:
        super().__init__(
            model_id=model_id,
            # LM Studio accepts any key; use provided or fall back to dummy
            api_key=api_key or _LMS_DUMMY_KEY,
            api_base=api_base or _DEFAULT_LMS_BASE,
            max_new_tokens=max_new_tokens,
        )

    @property
    def backend_name(self) -> str:
        return "lmstudio"
