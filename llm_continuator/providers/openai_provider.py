"""OpenAI / OpenAI-compatible provider adapter."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import BaseProvider, GenerationResult


class OpenAIProvider(BaseProvider):
    """Adapter for the OpenAI Chat Completions API (and compatible APIs).

    Parameters
    ----------
    model:
        Model identifier, e.g. ``"gpt-4o"`` or ``"gpt-3.5-turbo"``.
    api_key:
        OpenAI API key.  Falls back to the ``OPENAI_API_KEY`` environment
        variable when *None*.
    base_url:
        Override for OpenAI-compatible endpoints (e.g. local vLLM servers).
    client:
        Pre-built ``openai.OpenAI`` instance.  Useful for testing.
    default_params:
        Extra keyword arguments forwarded to every ``chat.completions.create``
        call (e.g. ``{"temperature": 0.2}``).
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        client: Any = None,
        default_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.default_params: Dict[str, Any] = default_params or {}

        if client is not None:
            self._client = client
        else:
            try:
                import openai  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "openai package is required for OpenAIProvider. "
                    "Install it with: pip install openai"
                ) from exc
            kwargs: Dict[str, Any] = {}
            if api_key:
                kwargs["api_key"] = api_key
            if base_url:
                kwargs["base_url"] = base_url
            self._client = openai.OpenAI(**kwargs)

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int,
        **kwargs: Any,
    ) -> GenerationResult:
        params = {**self.default_params, **kwargs}
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens,
            **params,
        )
        choice = response.choices[0]
        return GenerationResult(
            text=choice.message.content or "",
            finish_reason=choice.finish_reason,
            raw=response,
        )
