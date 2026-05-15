"""Output stitching strategies — combine partial LLM generations."""

from __future__ import annotations

import re
from typing import List


def _longest_common_suffix_prefix(a: str, b: str, max_window: int = 200) -> int:
    """Return the length of the longest suffix of *a* that is a prefix of *b*.

    Checked only up to *max_window* characters to stay O(n) in practice.
    """
    # Compare at most max_window chars from end of a vs start of b.
    window = min(max_window, len(a), len(b))
    for length in range(window, 0, -1):
        if a[-length:] == b[:length]:
            return length
    return 0


def concat_stitch(segments: List[str]) -> str:
    """Join segments with no overlap removal — fastest but may duplicate words."""
    return "".join(segments)


def smart_stitch(segments: List[str]) -> str:
    """Join segments, removing any overlapping text at segment boundaries."""
    if not segments:
        return ""
    result = segments[0]
    for segment in segments[1:]:
        overlap = _longest_common_suffix_prefix(result, segment)
        result += segment[overlap:]
    return result


def code_aware_stitch(segments: List[str]) -> str:
    """Like smart_stitch but additionally closes open code fences between joins."""
    if not segments:
        return ""
    result = segments[0]
    for segment in segments[1:]:
        # If the accumulated result has an open fence, close it before appending.
        if result.count("```") % 2 != 0:
            # Detect language specifier of the open fence if present.
            open_fence_match = re.search(r"```(\w*)\s*\n", result)
            lang = open_fence_match.group(1) if open_fence_match else ""
            result = result.rstrip() + "\n```\n"
            # If the continuation starts with a fence opener, skip it.
            stripped = segment.lstrip()
            if stripped.startswith(f"```{lang}"):
                segment = stripped[len(f"```{lang}"):]
                if segment.startswith("\n"):
                    segment = segment[1:]
        overlap = _longest_common_suffix_prefix(result, segment)
        result += segment[overlap:]
    return result


STRATEGIES = {
    "concat": concat_stitch,
    "smart": smart_stitch,
    "code_aware": code_aware_stitch,
}


def stitch(segments: List[str], strategy: str = "smart") -> str:
    """Dispatch to the requested stitching strategy.

    Parameters
    ----------
    segments:
        Ordered list of partial outputs to combine.
    strategy:
        One of ``"concat"``, ``"smart"``, ``"code_aware"``.
    """
    fn = STRATEGIES.get(strategy)
    if fn is None:
        raise ValueError(
            f"Unknown stitching strategy: {strategy!r}. "
            f"Choose from: {list(STRATEGIES)}"
        )
    return fn(segments)
