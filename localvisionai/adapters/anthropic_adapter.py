"""Anthropic backend adapter — Claude vision models.

Supports all Claude models with vision capability:
    claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5

Requires: pip install localvisionai[anthropic]
API key:  set ANTHROPIC_API_KEY env var, or pass api_key= to the constructor.
"""

from __future__ import annotations

import os
from typing import AsyncGenerator, Optional

from PIL import Image

from localvisionai.exceptions import ModelInferenceError, ModelNotFoundError
from localvisionai.utils.image import encode_to_base64, resize_frame, to_rgb
from localvisionai.utils.logging import get_logger
from .base import AbstractModelAdapter

logger = get_logger(__name__)

_DEFAULT_MODEL = "claude-sonnet-4-6"
_PREFERRED_RES = (512, 512)


class AnthropicAdapter(AbstractModelAdapter):
    """
    Vision model adapter for Anthropic's Claude API.

    Frames are encoded as JPEG base64 and passed via the Messages API
    image content block. Responses stream token-by-token.

    Supported models (latest):
        claude-opus-4-6        — most capable
        claude-sonnet-4-6      — best speed/quality balance (default)
        claude-haiku-4-5       — fastest, lowest cost

    Usage:
        adapter = AnthropicAdapter("claude-sonnet-4-6")  # key from ANTHROPIC_API_KEY
        await adapter.load()
        async for token in adapter.infer(frame, "Describe this frame."):
            print(token, end="", flush=True)
    """

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL,
        api_key: Optional[str] = None,
        max_new_tokens: int = 512,
    ) -> None:
        self._model_id = model_id
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._max_new_tokens = max_new_tokens
        self._client = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def backend_name(self) -> str:
        return "anthropic"

    @property
    def preferred_resolution(self):
        return _PREFERRED_RES

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def load(self) -> None:
        """Initialise the AsyncAnthropic client and verify the API key is present."""
        try:
            import anthropic
        except ImportError:
            raise ModelNotFoundError(
                "The 'anthropic' package is not installed. "
                "Install it with: pip install localvisionai[anthropic]"
            )

        if not self._api_key:
            raise ModelNotFoundError(
                "No Anthropic API key found. Set the ANTHROPIC_API_KEY environment variable "
                "or pass api_key= when constructing the adapter."
            )

        self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        logger.info(f"Anthropic adapter ready — model={self._model_id}")

    async def unload(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _encode_frame(self, frame: Image.Image) -> str:
        frame = to_rgb(resize_frame(frame, self.preferred_resolution))
        return encode_to_base64(frame, format="JPEG", quality=85)

    def _build_image_block(self, frame: Image.Image) -> dict:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": self._encode_frame(frame),
            },
        }

    async def infer(
        self,
        frame: Image.Image,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        if self._client is None:
            raise ModelInferenceError("Adapter not loaded. Call load() first.")

        content = [
            self._build_image_block(frame),
            {"type": "text", "text": prompt},
        ]

        kwargs: dict = {
            "model": self._model_id,
            "max_tokens": self._max_new_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            raise ModelInferenceError(
                f"Anthropic inference failed for model '{self._model_id}': {e}"
            ) from e

    async def infer_multi(
        self,
        frames: list[Image.Image],
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Send multiple frames as separate image blocks in a single user message."""
        if self._client is None:
            raise ModelInferenceError("Adapter not loaded. Call load() first.")

        content = [self._build_image_block(f) for f in frames]
        content.append({"type": "text", "text": prompt})

        kwargs: dict = {
            "model": self._model_id,
            "max_tokens": self._max_new_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            raise ModelInferenceError(
                f"Anthropic multi-frame inference failed for model '{self._model_id}': {e}"
            ) from e

    @property
    def supports_multi_frame(self) -> bool:
        return True
