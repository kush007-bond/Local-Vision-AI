"""Console output handler — prints timestamped descriptions using rich."""

from __future__ import annotations

from localvisionai.adapters.base import InferenceResult
from localvisionai.utils.timing import format_timestamp
from .base import AbstractOutputHandler


class ConsoleOutput(AbstractOutputHandler):
    """
    Prints each inference result to the terminal with rich formatting.

    Output format:
        [00:00:05.000] [gemma3/ollama] A person sits down at a desk and opens a laptop.
        (1240ms | 11 tokens)
    """

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self._console = None

    async def open(self, job_id: str) -> None:
        from rich.console import Console
        self._console = Console(highlight=False)
        self._console.rule(f"[bold cyan]LocalVisionAI — Job {job_id}")

    async def handle(self, result: InferenceResult) -> None:
        if self._console is None:
            from rich.console import Console
            self._console = Console(highlight=False)

        ts = format_timestamp(result.timestamp)
        model_tag = f"[dim]{result.model_id}/{result.backend}[/dim]"

        self._console.print(
            f"[bold green][{ts}][/bold green] {model_tag} {result.description}"
        )

        if self.verbose:
            self._console.print(
                f"  [dim]({result.latency_ms:.0f}ms | {result.token_count} tokens)[/dim]"
            )

    async def close(self) -> None:
        if self._console:
            self._console.rule("[bold cyan]Pipeline complete")
