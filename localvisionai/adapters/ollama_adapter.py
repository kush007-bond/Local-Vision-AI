"""Ollama backend adapter — uses the official Ollama Python SDK."""

from __future__ import annotations

import time
from typing import AsyncGenerator, Optional

from PIL import Image

from localvisionai.exceptions import ModelInferenceError, ModelNotFoundError
from localvisionai.utils.image import encode_to_base64, resize_frame, to_rgb
from localvisionai.utils.logging import get_logger
from .base import AbstractModelAdapter

logger = get_logger(__name__)


class OllamaAdapter(AbstractModelAdapter):
    """
    Vision model adapter for Ollama-served models.

    Supports any vision model available in Ollama's library:
    gemma3, qwen2-vl, llava, llava-llama3, moondream, bakllava, etc.

    The Ollama server must be running (default: http://localhost:11434).
    Models must be pulled beforehand: `ollama pull gemma3`
    """

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
    ) -> None:
        self._model = model
        self._host = host
        self._client = None

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def backend_name(self) -> str:
        return "ollama"

    @property
    def preferred_resolution(self):
        return (448, 448)

    async def load(self) -> None:
        """Verify the model is available in Ollama. Does not load weights (Ollama manages that)."""
        try:
            import ollama
        except ImportError:
            raise ModelNotFoundError(
                "The 'ollama' package is not installed. "
                "Install it with: pip install localvisionai[ollama]"
            )

        self._client = ollama.AsyncClient(host=self._host)

        try:
            response = await self._client.list()
            available = [m.model for m in response.models]
            # Ollama sometimes appends :latest
            aliases = {name.split(":")[0] for name in available} | set(available)
            model_base = self._model.split(":")[0]

            if model_base not in aliases and self._model not in aliases:
                raise ModelNotFoundError(
                    f"Model '{self._model}' not found in Ollama.\n"
                    f"Available models: {sorted(available)}\n"
                    f"Pull it with: ollama pull {self._model}"
                )
            logger.info(f"Ollama model verified: {self._model} @ {self._host}")
        except ModelNotFoundError:
            raise
        except Exception as e:
            raise ModelNotFoundError(
                f"Cannot connect to Ollama at {self._host}: {e}\n"
                "Make sure Ollama is running: ollama serve"
            ) from e

    async def unload(self) -> None:
        """Ollama manages its own model lifecycle — nothing to do here."""
        self._client = None

    def _encode_frame(self, frame: Image.Image) -> str:
        """Resize to preferred resolution and encode as base64 JPEG."""
        frame = to_rgb(frame)
        frame = resize_frame(frame, self.preferred_resolution)
        return encode_to_base64(frame, format="JPEG", quality=85)

    async def infer(
        self,
        frame: Image.Image,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        if self._client is None:
            raise ModelInferenceError("Adapter not loaded. Call load() first.")

        img_b64 = self._encode_frame(frame)
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": prompt,
            "images": [img_b64],
        })

        try:
            async for chunk in await self._client.chat(
                model=self._model,
                messages=messages,
                stream=True,
            ):
                token = chunk.message.content
                if token:
                    yield token
        except Exception as e:
            raise ModelInferenceError(
                f"Ollama inference failed for model '{self._model}': {e}"
            ) from e

    async def infer_multi(
        self,
        frames: list[Image.Image],
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Ollama supports multiple images in one message — pass all frames."""
        if self._client is None:
            raise ModelInferenceError("Adapter not loaded. Call load() first.")

        encoded = [self._encode_frame(f) for f in frames]
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": prompt,
            "images": encoded,
        })

        try:
            async for chunk in await self._client.chat(
                model=self._model,
                messages=messages,
                stream=True,
            ):
                token = chunk.message.content
                if token:
                    yield token
        except Exception as e:
            raise ModelInferenceError(
                f"Ollama multi-frame inference failed for model '{self._model}': {e}"
            ) from e

    @property
    def supports_multi_frame(self) -> bool:
        return True  # Ollama supports multiple images in one message
