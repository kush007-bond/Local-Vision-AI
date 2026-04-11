"""OpenAI backend adapter — GPT-4o and compatible vision models.

Supports any OpenAI vision model: gpt-4o, gpt-4o-mini, gpt-4-turbo.
Also works with any OpenAI-compatible endpoint by setting api_base.

Requires: pip install localvisionai[openai]
API key:  set OPENAI_API_KEY env var, or pass api_key= to the constructor.
"""

from __future__ import annotations

import base64
import os
from typing import TYPE_CHECKING, AsyncGenerator, Optional

from PIL import Image

from localvisionai.exceptions import ModelInferenceError, ModelNotFoundError
from localvisionai.utils.image import encode_to_base64, resize_frame, to_rgb
from localvisionai.utils.logging import get_logger
from .base import AbstractModelAdapter

if TYPE_CHECKING:
    from localvisionai.audio.base import AudioChunk

logger = get_logger(__name__)

_DEFAULT_MODEL = "gpt-4o"
_PREFERRED_RES = (512, 512)


class OpenAIAdapter(AbstractModelAdapter):
    """
    Vision model adapter for OpenAI's API (and compatible endpoints).

    Frame encoding: JPEG base64, sent as a data-URI in the image_url content block.

    Supported models:
        gpt-4o            — best quality, fastest
        gpt-4o-mini       — cheaper, still excellent vision
        gpt-4-turbo       — older but capable

    Usage:
        adapter = OpenAIAdapter("gpt-4o")   # key from OPENAI_API_KEY env var
        await adapter.load()
        async for token in adapter.infer(frame, "Describe this frame."):
            print(token, end="", flush=True)
    """

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_new_tokens: int = 512,
    ) -> None:
        self._model_id = model_id
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._api_base = api_base
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
        return "openai"

    @property
    def preferred_resolution(self):
        return _PREFERRED_RES

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def load(self) -> None:
        """Initialise the AsyncOpenAI client and verify the API key is present."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ModelNotFoundError(
                "The 'openai' package is not installed. "
                "Install it with: pip install localvisionai[openai]"
            )

        if not self._api_key:
            raise ModelNotFoundError(
                "No OpenAI API key found. Set the OPENAI_API_KEY environment variable "
                "or pass api_key= when constructing the adapter."
            )

        kwargs: dict = {"api_key": self._api_key}
        if self._api_base:
            kwargs["base_url"] = self._api_base

        self._client = AsyncOpenAI(**kwargs)
        logger.info(
            f"OpenAI adapter ready — model={self._model_id}"
            + (f" base_url={self._api_base}" if self._api_base else "")
        )

    async def unload(self) -> None:
        self._client = None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _encode_frame(self, frame: Image.Image) -> str:
        frame = to_rgb(resize_frame(frame, self.preferred_resolution))
        return encode_to_base64(frame, format="JPEG", quality=85)

    async def infer(
        self,
        frame: Image.Image,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        if self._client is None:
            raise ModelInferenceError("Adapter not loaded. Call load() first.")

        b64 = self._encode_frame(frame)
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "auto"},
                },
                {"type": "text", "text": prompt},
            ],
        })

        try:
            stream = await self._client.chat.completions.create(
                model=self._model_id,
                messages=messages,
                max_tokens=self._max_new_tokens,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
        except Exception as e:
            raise ModelInferenceError(
                f"OpenAI inference failed for model '{self._model_id}': {e}"
            ) from e

    async def infer_multi(
        self,
        frames: list[Image.Image],
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Send multiple frames as separate image_url blocks in a single message."""
        if self._client is None:
            raise ModelInferenceError("Adapter not loaded. Call load() first.")

        content: list = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{self._encode_frame(f)}",
                    "detail": "auto",
                },
            }
            for f in frames
        ]
        content.append({"type": "text", "text": prompt})

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        try:
            stream = await self._client.chat.completions.create(
                model=self._model_id,
                messages=messages,
                max_tokens=self._max_new_tokens,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
        except Exception as e:
            raise ModelInferenceError(
                f"OpenAI multi-frame inference failed for model '{self._model_id}': {e}"
            ) from e

    @property
    def supports_multi_frame(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Native audio
    # ------------------------------------------------------------------

    @property
    def supports_audio(self) -> bool:
        """GPT-4o and gpt-4o-audio-preview accept base64 WAV input."""
        mid = self._model_id.lower()
        return "audio" in mid or mid.startswith("gpt-4o")

    async def infer_with_audio(
        self,
        frame: Image.Image,
        audio: "AudioChunk",
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Send frame + raw audio (base64 WAV) in a single chat.completions call."""
        if self._client is None:
            raise ModelInferenceError("Adapter not loaded. Call load() first.")

        # Local import to avoid a hard dependency when audio is not used.
        from localvisionai.audio.segmenter import AudioSegmenter

        b64_img = self._encode_frame(frame)
        wav_bytes = AudioSegmenter.chunk_to_wav_bytes(audio)

        content: list = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_img}",
                    "detail": "auto",
                },
            },
        ]
        if wav_bytes:
            b64_audio = base64.b64encode(wav_bytes).decode("ascii")
            content.append(
                {
                    "type": "input_audio",
                    "input_audio": {"data": b64_audio, "format": "wav"},
                }
            )
        content.append({"type": "text", "text": prompt})

        messages: list = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        try:
            stream = await self._client.chat.completions.create(
                model=self._model_id,
                messages=messages,
                max_tokens=self._max_new_tokens,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
        except Exception as e:
            raise ModelInferenceError(
                f"OpenAI audio inference failed for model '{self._model_id}': {e}"
            ) from e
