"""Google Gemini backend adapter — vision models via google-generativeai SDK.

Supports Gemini models with vision: gemini-2.0-flash, gemini-1.5-pro,
gemini-1.5-flash, gemini-1.5-flash-8b.

Requires: pip install localvisionai[gemini]
API key:  set GOOGLE_API_KEY env var, or pass api_key= to the constructor.
"""

from __future__ import annotations

import os
from typing import AsyncGenerator, Optional

from PIL import Image

from localvisionai.exceptions import ModelInferenceError, ModelNotFoundError
from localvisionai.utils.image import resize_frame, to_rgb
from localvisionai.utils.logging import get_logger
from .base import AbstractModelAdapter

logger = get_logger(__name__)

_DEFAULT_MODEL = "gemini-2.0-flash"
_PREFERRED_RES = (768, 768)


class GeminiAdapter(AbstractModelAdapter):
    """
    Vision model adapter for Google's Gemini API.

    Frames are passed as PIL images directly to the SDK (no manual encoding
    needed — the SDK handles JPEG conversion internally).

    Supported models (vision-capable):
        gemini-2.0-flash       — fastest, best for real-time (default)
        gemini-1.5-pro         — most capable, larger context
        gemini-1.5-flash       — speed/quality balance
        gemini-1.5-flash-8b    — smallest, lowest cost

    Usage:
        adapter = GeminiAdapter("gemini-2.0-flash")  # key from GOOGLE_API_KEY
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
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self._max_new_tokens = max_new_tokens
        self._model = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def backend_name(self) -> str:
        return "gemini"

    @property
    def preferred_resolution(self):
        return _PREFERRED_RES

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def load(self) -> None:
        """Configure the genai SDK and instantiate the GenerativeModel."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise ModelNotFoundError(
                "The 'google-generativeai' package is not installed. "
                "Install it with: pip install localvisionai[gemini]"
            )

        if not self._api_key:
            raise ModelNotFoundError(
                "No Google API key found. Set the GOOGLE_API_KEY environment variable "
                "or pass api_key= when constructing the adapter."
            )

        genai.configure(api_key=self._api_key)
        self._model = genai.GenerativeModel(model_name=self._model_id)
        logger.info(f"Gemini adapter ready — model={self._model_id}")

    async def unload(self) -> None:
        self._model = None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _prepare_frame(self, frame: Image.Image) -> Image.Image:
        """Resize to preferred resolution, ensure RGB."""
        return to_rgb(resize_frame(frame, self.preferred_resolution))

    async def infer(
        self,
        frame: Image.Image,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        if self._model is None:
            raise ModelInferenceError("Adapter not loaded. Call load() first.")

        import google.generativeai as genai

        pil_frame = self._prepare_frame(frame)

        # System instruction requires re-creating the model with it set
        model = self._model
        if system_prompt:
            model = genai.GenerativeModel(
                model_name=self._model_id,
                system_instruction=system_prompt,
            )

        gen_config = genai.types.GenerationConfig(
            max_output_tokens=self._max_new_tokens,
        )

        try:
            response = await model.generate_content_async(
                [pil_frame, prompt],
                generation_config=gen_config,
                stream=True,
            )
            async for chunk in response:
                text = chunk.text if hasattr(chunk, "text") else None
                if text:
                    yield text
        except Exception as e:
            raise ModelInferenceError(
                f"Gemini inference failed for model '{self._model_id}': {e}"
            ) from e

    async def infer_multi(
        self,
        frames: list[Image.Image],
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Interleave all frames with the prompt in a single content list."""
        if self._model is None:
            raise ModelInferenceError("Adapter not loaded. Call load() first.")

        import google.generativeai as genai

        model = self._model
        if system_prompt:
            model = genai.GenerativeModel(
                model_name=self._model_id,
                system_instruction=system_prompt,
            )

        gen_config = genai.types.GenerationConfig(
            max_output_tokens=self._max_new_tokens,
        )

        # Build content list: [frame1, frame2, ..., prompt]
        content = [self._prepare_frame(f) for f in frames]
        content.append(prompt)

        try:
            response = await model.generate_content_async(
                content,
                generation_config=gen_config,
                stream=True,
            )
            async for chunk in response:
                text = chunk.text if hasattr(chunk, "text") else None
                if text:
                    yield text
        except Exception as e:
            raise ModelInferenceError(
                f"Gemini multi-frame inference failed for model '{self._model_id}': {e}"
            ) from e

    @property
    def supports_multi_frame(self) -> bool:
        return True
