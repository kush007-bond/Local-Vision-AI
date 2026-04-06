"""Abstract model adapter interface and InferenceResult data class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional, Tuple

from PIL import Image


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

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "description": self.description,
            "model_id": self.model_id,
            "backend": self.backend,
            "latency_ms": round(self.latency_ms, 1),
            "token_count": self.token_count,
        }


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
    def preferred_resolution(self) -> Tuple[int, int]:
        """
        Return (width, height) that this model performs best at.
        The pipeline uses this to resize frames before inference.
        """
        return (448, 448)
