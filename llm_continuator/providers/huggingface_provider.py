"""HuggingFace Inference API (serverless) provider adapter."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import BaseProvider, GenerationResult


class HuggingFaceProvider(BaseProvider):
    """Adapter for the HuggingFace Inference API (chat completions endpoint).

    Uses the ``huggingface_hub`` InferenceClient which supports the
    OpenAI-compatible ``/v1/chat/completions`` endpoint available on many
    HF-hosted models.

    Parameters
    ----------
    model:
        HuggingFace model repo ID, e.g. ``"meta-llama/Llama-3-8B-Instruct"``.
    api_key:
        HuggingFace API token.  Falls back to the ``HUGGINGFACE_API_KEY``
        or ``HF_TOKEN`` environment variable.
    client:
        Pre-built ``huggingface_hub.InferenceClient`` instance.  Useful for
        testing.
    default_params:
        Extra keyword arguments forwarded to every ``chat_completion`` call.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        client: Any = None,
        default_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.default_params: Dict[str, Any] = default_params or {}

        if client is not None:
            self._client = client
        else:
            try:
                from huggingface_hub import InferenceClient  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "huggingface_hub package is required for HuggingFaceProvider. "
                    "Install it with: pip install huggingface_hub"
                ) from exc
            self._client = InferenceClient(
                model=model,
                **({"token": api_key} if api_key else {}),
            )

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int,
        **kwargs: Any,
    ) -> GenerationResult:
        params = {**self.default_params, **kwargs}
        response = self._client.chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            **params,
        )
        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None) or "unknown"
        # Normalise "eos_token" → "stop", "length" stays as-is.
        if finish_reason == "eos_token":
            finish_reason = "stop"
        return GenerationResult(
            text=choice.message.content or "",
            finish_reason=finish_reason,
            raw=response,
        )
