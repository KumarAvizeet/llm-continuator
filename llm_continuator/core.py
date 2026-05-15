"""Core Continuator — orchestrates retry logic and output stitching."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .config import ContinuatorConfig
from .detectors import TruncationDetector
from .exceptions import MaxRetriesExceededError
from .providers.base import BaseProvider, GenerationResult
from .stitchers import stitch

logger = logging.getLogger(__name__)


class CompletionResult:
    """Final result returned by :meth:`Continuator.complete`.

    Attributes
    ----------
    text:
        The fully assembled response text.
    attempts:
        Total number of LLM calls made (1 = no retries needed).
    segments:
        Individual raw generation segments in order.
    was_truncated:
        *True* if at least one retry was needed.
    last_finish_reason:
        The finish_reason of the final segment.
    """

    __slots__ = ("text", "attempts", "segments", "was_truncated", "last_finish_reason")

    def __init__(
        self,
        text: str,
        attempts: int,
        segments: List[str],
        was_truncated: bool,
        last_finish_reason: Optional[str],
    ) -> None:
        self.text = text
        self.attempts = attempts
        self.segments = segments
        self.was_truncated = was_truncated
        self.last_finish_reason = last_finish_reason

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"CompletionResult(attempts={self.attempts}, "
            f"was_truncated={self.was_truncated}, "
            f"text_len={len(self.text)})"
        )


class Continuator:
    """High-level wrapper that ensures a complete LLM response.

    The Continuator calls the provider, checks whether the output was
    truncated, and — if so — issues continuation requests until the
    response is complete or ``config.max_retries`` is exhausted.

    Parameters
    ----------
    provider:
        Any :class:`~llm_continuator.providers.base.BaseProvider` subclass.
    config:
        :class:`~llm_continuator.config.ContinuatorConfig` instance.
        Defaults are used when omitted.

    Examples
    --------
    >>> from llm_continuator import Continuator, ContinuatorConfig
    >>> from llm_continuator.providers import OpenAIProvider
    >>>
    >>> provider = OpenAIProvider(model="gpt-4o")
    >>> config = ContinuatorConfig(max_retries=5, max_new_tokens=1024)
    >>> cont = Continuator(provider, config)
    >>> result = cont.complete([{"role": "user", "content": "Write a long essay"}])
    >>> print(result.text)
    """

    def __init__(
        self,
        provider: BaseProvider,
        config: Optional[ContinuatorConfig] = None,
    ) -> None:
        self.provider = provider
        self.config = config or ContinuatorConfig()
        self._detector = TruncationDetector(self.config.truncation_sensitivity)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete(
        self,
        messages: List[Dict[str, str]],
        **generation_kwargs: Any,
    ) -> CompletionResult:
        """Generate a complete response, retrying if output is truncated.

        Parameters
        ----------
        messages:
            Chat message list, e.g.
            ``[{"role": "user", "content": "..."}]``.
        **generation_kwargs:
            Extra keyword arguments forwarded to each provider call
            (e.g. ``temperature=0.7``).

        Returns
        -------
        CompletionResult
            Contains the assembled text plus diagnostic metadata.

        Raises
        ------
        MaxRetriesExceededError
            When ``config.max_retries`` attempts are exhausted and
            ``config.raise_on_max_retries`` is *True*.
        """
        cfg = self.config
        segments: List[str] = []
        current_messages = list(messages)
        was_truncated = False
        last_result: Optional[GenerationResult] = None

        for attempt in range(1, cfg.max_retries + 2):  # +1 for the initial call
            logger.debug("llm-continuator: attempt %d", attempt)

            last_result = self.provider.generate(
                current_messages,
                max_new_tokens=cfg.max_new_tokens,
                **generation_kwargs,
            )
            segments.append(last_result.text)

            truncated = self._detector.is_truncated(
                last_result.text, last_result.finish_reason
            )

            if not truncated:
                logger.debug(
                    "llm-continuator: response complete on attempt %d", attempt
                )
                break

            # We are out of retry budget.
            if attempt > cfg.max_retries:
                logger.warning(
                    "llm-continuator: still truncated after %d retries", cfg.max_retries
                )
                assembled = stitch(segments, cfg.stitching_strategy)
                if cfg.raise_on_max_retries:
                    raise MaxRetriesExceededError(
                        attempts=attempt, partial_output=assembled
                    )
                break

            was_truncated = True
            logger.debug(
                "llm-continuator: truncation detected (finish_reason=%r), "
                "issuing continuation request %d/%d",
                last_result.finish_reason,
                attempt,
                cfg.max_retries,
            )
            current_messages = self._build_continuation_messages(
                original_messages=messages,
                accumulated=stitch(segments, cfg.stitching_strategy),
            )

        assembled = stitch(segments, cfg.stitching_strategy)
        return CompletionResult(
            text=assembled,
            attempts=len(segments),
            segments=segments,
            was_truncated=was_truncated,
            last_finish_reason=last_result.finish_reason if last_result else None,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_continuation_messages(
        self,
        original_messages: List[Dict[str, str]],
        accumulated: str,
    ) -> List[Dict[str, str]]:
        """Construct a message list that asks the model to continue its output."""
        cfg = self.config
        tail = accumulated[-cfg.continuation_window:]
        continuation_prompt = cfg.continuation_prompt_template.format(partial=tail)

        # Start from the original conversation, append the partial assistant
        # turn, and then inject the continuation instruction as a new user turn.
        return [
            *original_messages,
            {"role": "assistant", "content": accumulated},
            {"role": "user", "content": continuation_prompt},
        ]
