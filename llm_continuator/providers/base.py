"""Abstract base for LLM provider adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class GenerationResult:
    """Unified response object returned by every provider."""

    __slots__ = ("text", "finish_reason", "raw")

    def __init__(
        self,
        text: str,
        finish_reason: Optional[str],
        raw: Any = None,
    ) -> None:
        self.text = text
        self.finish_reason = finish_reason
        # Original provider response for introspection.
        self.raw = raw

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"GenerationResult(finish_reason={self.finish_reason!r}, "
            f"text_len={len(self.text)})"
        )


class BaseProvider(ABC):
    """Interface every provider adapter must implement."""

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int,
        **kwargs: Any,
    ) -> GenerationResult:
        """Call the underlying LLM and return a :class:`GenerationResult`.

        Parameters
        ----------
        messages:
            Chat-style message list, e.g.
            ``[{"role": "user", "content": "Hello"}]``.
        max_new_tokens:
            Hard upper bound on tokens produced in this single call.
        **kwargs:
            Provider-specific extra parameters (temperature, top_p, …).
        """
