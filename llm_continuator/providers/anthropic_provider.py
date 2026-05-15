"""Anthropic Messages API provider adapter."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import BaseProvider, GenerationResult

# Map Anthropic stop reasons to a canonical finish_reason vocabulary.
_STOP_REASON_MAP = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_use",
}


class AnthropicProvider(BaseProvider):
    """Adapter for the Anthropic Messages API.

    Parameters
    ----------
    model:
        Model ID, e.g. ``"claude-sonnet-4-6"`` or ``"claude-opus-4-7"``.
    api_key:
        Anthropic API key.  Falls back to the ``ANTHROPIC_API_KEY``
        environment variable when *None*.
    client:
        Pre-built ``anthropic.Anthropic`` instance.  Useful for testing.
    system_prompt:
        Optional system prompt prepended to every request.
    default_params:
        Extra keyword arguments forwarded to every ``messages.create`` call.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
        client: Any = None,
        system_prompt: Optional[str] = None,
        default_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.default_params: Dict[str, Any] = default_params or {}

        if client is not None:
            self._client = client
        else:
            try:
                import anthropic  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "anthropic package is required for AnthropicProvider. "
                    "Install it with: pip install anthropic"
                ) from exc
            self._client = anthropic.Anthropic(
                **({"api_key": api_key} if api_key else {})
            )

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int,
        **kwargs: Any,
    ) -> GenerationResult:
        params: Dict[str, Any] = {**self.default_params, **kwargs}
        if self.system_prompt:
            params.setdefault("system", self.system_prompt)

        response = self._client.messages.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens,
            **params,
        )
        text = "".join(
            block.text for block in response.content if hasattr(block, "text")
        )
        finish_reason = _STOP_REASON_MAP.get(
            response.stop_reason or "", response.stop_reason or "unknown"
        )
        return GenerationResult(text=text, finish_reason=finish_reason, raw=response)
