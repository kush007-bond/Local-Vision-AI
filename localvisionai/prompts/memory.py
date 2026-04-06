"""Sliding window context memory for multi-frame temporal awareness."""

from __future__ import annotations

from collections import deque
from typing import Optional


class ContextWindow:
    """
    Maintains a rolling summary of recent frame descriptions.

    When context_mode='sliding_window', this is injected into each new
    prompt so single-image models get temporal awareness of prior frames.

    Token budget is approximate (splits on whitespace).
    """

    def __init__(self, max_tokens: int = 200) -> None:
        self.max_tokens = max_tokens
        self._entries: deque[str] = deque()
        self._total_tokens: int = 0

    def update(self, description: str) -> None:
        """Add a new frame description to the context window."""
        if not description.strip():
            return

        tokens = len(description.split())
        self._entries.append(description)
        self._total_tokens += tokens

        # Evict oldest entries if over budget
        while self._total_tokens > self.max_tokens and self._entries:
            oldest = self._entries.popleft()
            self._total_tokens -= len(oldest.split())

    def get_summary(self) -> Optional[str]:
        """Return the rolling context as a single string, or None if empty."""
        if not self._entries:
            return None
        return " ".join(self._entries)

    def reset(self) -> None:
        """Clear all context. Called at the start of each new video."""
        self._entries.clear()
        self._total_tokens = 0

    def __len__(self) -> int:
        return len(self._entries)
