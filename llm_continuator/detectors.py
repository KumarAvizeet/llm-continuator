"""Truncation detectors — decide whether an LLM response was cut off."""

from __future__ import annotations

import re
from typing import Optional


def _open_code_fence(text: str) -> bool:
    """Return True if there is an unclosed triple-backtick code fence."""
    return text.count("```") % 2 != 0


def _open_brackets(text: str) -> bool:
    """Return True if any bracket type is left unbalanced (net-open)."""
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack: list[str] = []
    in_string: Optional[str] = None
    i = 0
    while i < len(text):
        ch = text[i]
        # Simple string tracking (not perfect, but good enough for heuristics).
        if in_string:
            if ch == "\\" and i + 1 < len(text):
                i += 2
                continue
            if ch == in_string:
                in_string = None
        elif ch in ('"', "'"):
            in_string = ch
        elif ch in pairs:
            stack.append(pairs[ch])
        elif stack and ch == stack[-1]:
            stack.pop()
        i += 1
    return bool(stack)


def _ends_mid_sentence(text: str) -> bool:
    """Heuristic: text ends without a sentence-terminating punctuation."""
    stripped = text.rstrip()
    if not stripped:
        return False
    last_char = stripped[-1]
    # Acceptable endings: . ! ? " ' ` ) ] } : (for code) or a digit (step N)
    return last_char not in {".", "!", "?", '"', "'", "`", ")", "]", "}", ":", ";"}


def _ends_mid_list(text: str) -> bool:
    """Heuristic: last non-empty line ends with a comma or starts a new bullet."""
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    last = lines[-1]
    if last.endswith(","):
        return True
    # Dangling list marker with no content
    if re.match(r"^\s*[-*•]\s*$", last):
        return True
    # Numbered list item that opens but provides nothing
    if re.match(r"^\s*\d+\.\s*$", last):
        return True
    return False


def _ends_mid_markdown_header(text: str) -> bool:
    """Last non-empty line is a markdown header with nothing after it."""
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    return bool(re.match(r"^#{1,6}\s+\S", lines[-1])) and len(lines) == 1


class TruncationDetector:
    """Evaluates whether a model output was prematurely truncated.

    Parameters
    ----------
    sensitivity:
        ``low``    → only trust API finish_reason == "length".
        ``medium`` → add syntactic checks (code fence, brackets, sentence end).
        ``high``   → additionally use heuristic patterns (mid-list, etc.).
    """

    def __init__(self, sensitivity: str = "medium") -> None:
        self.sensitivity = sensitivity

    def is_truncated(
        self,
        text: str,
        finish_reason: Optional[str] = None,
    ) -> bool:
        """Return *True* if *text* appears to be an incomplete response.

        Parameters
        ----------
        text:
            The raw text produced by the LLM.
        finish_reason:
            The finish/stop reason reported by the provider API, if any.
            Common values: ``"stop"``, ``"length"``, ``"end_turn"``,
            ``"max_tokens"``, ``None``.
        """
        # Explicit API signal — always authoritative.
        if finish_reason in {"length", "max_tokens", "max_new_tokens"}:
            return True

        if self.sensitivity == "low":
            return False

        # --- medium checks ---
        if _open_code_fence(text):
            return True
        if _ends_mid_sentence(text):
            return True
        # Unbalanced brackets are an expensive check; skip for very long texts.
        if len(text) < 8_000 and _open_brackets(text):
            return True

        if self.sensitivity == "medium":
            return False

        # --- high checks ---
        if _ends_mid_list(text):
            return True
        if _ends_mid_markdown_header(text):
            return True

        return False
