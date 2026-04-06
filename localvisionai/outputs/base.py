"""Abstract base class for output handlers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from localvisionai.adapters.base import InferenceResult


class AbstractOutputHandler(ABC):
    """
    Base class for all output handlers.

    Output handlers receive every InferenceResult and persist or display it.
    The pipeline fans out results to all registered handlers in sequence.

    Handlers must be async — if they do file I/O, use aiofiles or run in executor.
    """

    @abstractmethod
    async def handle(self, result: InferenceResult) -> None:
        """Process a single inference result."""
        ...

    async def open(self, job_id: str) -> None:
        """Called once when a job starts. Override for setup (open files, etc.)."""
        pass

    async def close(self) -> None:
        """Called once when a job finishes. Override for cleanup (flush, close files, etc.)."""
        pass
