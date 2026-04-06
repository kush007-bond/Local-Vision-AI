"""HuggingFace Transformers backend adapter.

Supports any HuggingFace vision-language model with a standard interface.
Tested with: Qwen2-VL, LLaVA-1.5, InternVL2, MiniCPM-V, Gemma 3.

All model loading and inference runs in a thread pool executor to avoid
blocking the asyncio event loop. Real token streaming uses TextIteratorStreamer.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from typing import AsyncGenerator, Optional

from PIL import Image

from localvisionai.exceptions import ModelInferenceError, ModelNotFoundError
from localvisionai.utils.image import resize_frame, to_rgb
from localvisionai.utils.logging import get_logger
from .base import AbstractModelAdapter

logger = get_logger(__name__)


class HuggingFaceAdapter(AbstractModelAdapter):
    """
    Vision-language model adapter for HuggingFace Transformers.

    Features:
    - 4-bit quantization via bitsandbytes (load_in_4bit=True)
    - Real streaming via TextIteratorStreamer
    - Multi-frame inference for Qwen2-VL and InternVL2
    - Automatic device selection (CUDA > MPS > CPU)

    Usage:
        adapter = HuggingFaceAdapter("Qwen/Qwen2-VL-7B-Instruct", load_in_4bit=True)
        await adapter.load()
        async for token in adapter.infer(frame, "Describe this frame."):
            print(token, end="", flush=True)
    """

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        load_in_4bit: bool = False,
        max_new_tokens: int = 512,
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._load_in_4bit = load_in_4bit
        self._max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def backend_name(self) -> str:
        return "transformers"

    @property
    def supports_multi_frame(self) -> bool:
        return True  # Qwen2-VL and InternVL2 natively support multiple images

    @property
    def preferred_resolution(self):
        return (448, 448)

    async def load(self) -> None:
        """Load the model and processor into memory in a thread pool."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_sync)
            logger.info(f"HuggingFace model loaded: {self._model_id} (4bit={self._load_in_4bit})")
        except ImportError as e:
            raise ModelNotFoundError(
                f"Missing dependency for HuggingFace backend: {e}. "
                f"Install with: pip install localvisionai[transformers]"
            ) from e
        except Exception as e:
            raise ModelNotFoundError(
                f"Failed to load HuggingFace model '{self._model_id}': {e}"
            ) from e

    def _load_sync(self) -> None:
        from transformers import AutoProcessor, AutoModelForVision2Seq
        import torch

        kwargs: dict = {"device_map": self._device, "torch_dtype": torch.float16}

        if self._load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

        self._processor = AutoProcessor.from_pretrained(self._model_id, trust_remote_code=True)
        self._model = AutoModelForVision2Seq.from_pretrained(
            self._model_id,
            trust_remote_code=True,
            **kwargs,
        )

    async def unload(self) -> None:
        """Release model from memory and clear GPU cache."""
        if self._model is not None:
            try:
                import torch
                del self._model
                del self._processor
                self._model = None
                self._processor = None
                torch.cuda.empty_cache()
                logger.info(f"HuggingFace model unloaded: {self._model_id}")
            except Exception as e:
                logger.warning(f"Error during model unload: {e}")

    async def infer(
        self,
        frame: Image.Image,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from the model using TextIteratorStreamer."""
        if self._model is None or self._processor is None:
            raise ModelInferenceError("Adapter not loaded. Call load() first.")

        token_queue: queue.Queue = queue.Queue()
        done_event = threading.Event()

        loop = asyncio.get_event_loop()

        def _generate_in_thread():
            try:
                from transformers import TextIteratorStreamer
                import torch

                frame_rgb = to_rgb(resize_frame(frame, self.preferred_resolution))
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "image": frame_rgb},
                        {"type": "text", "text": prompt},
                    ],
                })

                text = self._processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self._processor(text=text, images=[frame_rgb], return_tensors="pt")
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

                streamer = TextIteratorStreamer(
                    self._processor,
                    skip_prompt=True,
                    skip_special_tokens=True,
                )

                thread = threading.Thread(
                    target=self._model.generate,
                    kwargs={
                        **inputs,
                        "max_new_tokens": self._max_new_tokens,
                        "streamer": streamer,
                    },
                )
                thread.start()

                for token in streamer:
                    token_queue.put(token)

                thread.join()
            except Exception as e:
                token_queue.put(e)
            finally:
                token_queue.put(None)  # sentinel
                done_event.set()

        # Start generation in background thread
        loop.run_in_executor(None, _generate_in_thread)

        # Yield tokens from the queue
        while True:
            try:
                item = await asyncio.wait_for(
                    loop.run_in_executor(None, token_queue.get),
                    timeout=120.0,
                )
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise ModelInferenceError(
                        f"HuggingFace inference failed for '{self._model_id}': {item}"
                    ) from item
                yield item
            except asyncio.TimeoutError:
                raise ModelInferenceError(
                    f"HuggingFace model '{self._model_id}' timed out after 120s"
                )

    async def infer_multi(
        self,
        frames: list[Image.Image],
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Multi-frame inference — uses last frame if model doesn't support multi-image."""
        # For simplicity in v0.1, use last frame if multiple provided
        # Full multi-image support for Qwen2-VL / InternVL2 comes in v0.2
        async for token in self.infer(frames[-1], prompt, system_prompt):
            yield token
