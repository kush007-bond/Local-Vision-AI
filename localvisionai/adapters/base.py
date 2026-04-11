"""Abstract model adapter interface and InferenceResult data class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, AsyncGenerator, Optional, Tuple

from PIL import Image

if TYPE_CHECKING:
    from localvisionai.audio.base import AudioChunk


@dataclass
class InferenceResult:
    """The structured output of one model inference call."""

    timestamp: float                    # Seconds from video start
    description: str                    # Full assembled natural language description
    model_id: str                       # Canonical model identifier
    backend: str                        # Backend name: "ollama" | "transformers" | etc.
    latency_ms: float                   # End-to-end inference time in milliseconds
    token_count: int                    # Number of tokens generated
    raw_tokens: list[str] = field(default_factory=list)  # Individual tokens (if streamed)
    audio_transcript: Optional[str] = None  # Populated when transcription was used
    audio_mode: Optional[str] = None        # "native" | "transcribe" | None

    def to_dict(self) -> dict:
        payload = {
            "timestamp": self.timestamp,
            "description": self.description,
            "model_id": self.model_id,
            "backend": self.backend,
            "latency_ms": round(self.latency_ms, 1),
            "token_count": self.token_count,
        }
        if self.audio_mode is not None:
            payload["audio_mode"] = self.audio_mode
        if self.audio_transcript is not None:
            payload["audio_transcript"] = self.audio_transcript
        return payload


class AbstractModelAdapter(ABC):
    """
    Base class for all model backend adapters.

    Each backend (Ollama, HuggingFace, llama.cpp, MLX) implements this interface.
    The pipeline interacts with adapters exclusively through this interface,
    making backends fully swappable via config.

    Implementations must be safe to call concurrently if running multiple
    consumer tasks (though the default pipeline uses a single consumer).
    """

    @abstractmethod
    async def load(self) -> None:
        """
        Load model into memory. Called once at pipeline start.
        For backends like Ollama, this verifies availability without loading weights.
        For transformers/llama.cpp, this loads the model weights into RAM/VRAM.
        """
        ...

    @abstractmethod
    async def unload(self) -> None:
        """
        Release model from memory. Called on pipeline shutdown.
        Should free all GPU/CPU resources.
        """
        ...

    @abstractmethod
    async def infer(
        self,
        frame: Image.Image,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Async generator that yields tokens as they stream from the model.

        The caller is responsible for collecting tokens into a full description.
        Implementations should stream tokens as they're generated, not buffer them.

        Args:
            frame: The current video frame as a PIL Image (RGB).
            prompt: The user prompt / question to ask about the frame.
            system_prompt: Optional system prompt override.

        Yields:
            Individual tokens (strings) as they are generated.
        """
        ...

    @abstractmethod
    async def infer_multi(
        self,
        frames: list[Image.Image],
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Multi-frame inference for models that support multiple images per prompt.

        Models that don't natively support multi-frame should fall back to
        single-frame by using only the last frame in the list.

        Yields:
            Individual tokens (strings) as they are generated.
        """
        ...

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Return the canonical model identifier string."""
        ...

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the backend name (used in InferenceResult.backend)."""
        ...

    @property
    def supports_multi_frame(self) -> bool:
        """Override to True in adapters that natively support multiple images."""
        return False

    @property
    def supports_audio(self) -> bool:
        """Override to True in adapters that accept raw audio bytes alongside a frame."""
        return False

    async def infer_with_audio(
        self,
        frame: Image.Image,
        audio: "AudioChunk",
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream tokens while passing the audio chunk natively to the model.

        The default implementation raises NotImplementedError — the pipeline
        only dispatches to this method when `supports_audio` is True, so
        adapters that declare native audio support must override both.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement infer_with_audio(). "
            "Override it, or set supports_audio=False and use the transcription fallback."
        )
        # Make this a proper async generator even though the body raises —
        # keeps type-checkers and the AsyncGenerator protocol happy.
        if False:  # pragma: no cover
            yield ""

    @property
    def preferred_resolution(self) -> Tuple[int, int]:
        """
        Return (width, height) that this model performs best at.
        The pipeline uses this to resize frames before inference.
        """
        return (448, 448)
