"""Configuration dataclass for llm-continuator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .exceptions import ConfigurationError

StitchingStrategy = Literal["concat", "smart", "code_aware"]


@dataclass
class ContinuatorConfig:
    """All tuneable parameters for the Continuator.

    Parameters
    ----------
    max_retries:
        Maximum number of continuation attempts before giving up.
    max_new_tokens:
        Token budget passed to the provider on every call, including
        continuation calls.
    stitching_strategy:
        How partial generations are joined.
        - ``concat``      – simple string concatenation (fastest).
        - ``smart``       – trims overlapping tokens between segments.
        - ``code_aware``  – like *smart* but also repairs open code fences.
    continuation_prompt_template:
        Template used to ask the model to continue.  The placeholder
        ``{partial}`` is replaced with the last ``continuation_window``
        characters of the accumulated output.
    continuation_window:
        Number of trailing characters from the accumulated output to
        include in the continuation prompt so the model has context.
    truncation_sensitivity:
        How aggressively truncation is detected.
        ``low``    → only flag finish_reason == "length".
        ``medium`` → also flag syntactically open sentences / code.
        ``high``   → additionally flag heuristic patterns (trailing comma,
                     mid-list, etc.).
    raise_on_max_retries:
        If *True* (default), raise :class:`MaxRetriesExceededError` when
        retries are exhausted.  If *False*, return whatever has been
        accumulated so far.
    """

    max_retries: int = 5
    max_new_tokens: int = 1024
    stitching_strategy: StitchingStrategy = "smart"
    continuation_prompt_template: str = (
        "Continue exactly from where you stopped. "
        "Do not repeat any text. "
        "Here is the end of your previous response:\n\n...{partial}"
    )
    continuation_window: int = 300
    truncation_sensitivity: Literal["low", "medium", "high"] = "medium"
    raise_on_max_retries: bool = True

    def __post_init__(self) -> None:
        if self.max_retries < 1:
            raise ConfigurationError("max_retries must be >= 1")
        if self.max_new_tokens < 1:
            raise ConfigurationError("max_new_tokens must be >= 1")
        if self.continuation_window < 10:
            raise ConfigurationError("continuation_window must be >= 10")
        if self.stitching_strategy not in ("concat", "smart", "code_aware"):
            raise ConfigurationError(
                f"Unknown stitching_strategy: {self.stitching_strategy!r}. "
                "Choose from 'concat', 'smart', or 'code_aware'."
            )
        if self.truncation_sensitivity not in ("low", "medium", "high"):
            raise ConfigurationError(
                f"Unknown truncation_sensitivity: {self.truncation_sensitivity!r}"
            )
