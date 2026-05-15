"""Unit tests for provider adapters — all using mocked SDK clients."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from llm_continuator.providers.openai_provider import OpenAIProvider
from llm_continuator.providers.anthropic_provider import AnthropicProvider
from llm_continuator.providers.huggingface_provider import HuggingFaceProvider


# ---------------------------------------------------------------------------
# OpenAI Provider
# ---------------------------------------------------------------------------


def _make_openai_response(content: str, finish_reason: str = "stop"):
    """Build a minimal mock that looks like an OpenAI ChatCompletion response."""
    choice = SimpleNamespace(
        message=SimpleNamespace(content=content),
        finish_reason=finish_reason,
    )
    return SimpleNamespace(choices=[choice])


class TestOpenAIProvider:
    def _make_provider(self, content: str, finish_reason: str = "stop"):
        client = MagicMock()
        client.chat.completions.create.return_value = _make_openai_response(
            content, finish_reason
        )
        return OpenAIProvider(model="gpt-4o", client=client), client

    def test_generate_returns_text(self):
        provider, _ = self._make_provider("Hello!")
        result = provider.generate([{"role": "user", "content": "Hi"}], max_new_tokens=100)
        assert result.text == "Hello!"

    def test_finish_reason_forwarded(self):
        provider, _ = self._make_provider("text", "length")
        result = provider.generate([], max_new_tokens=10)
        assert result.finish_reason == "length"

    def test_max_new_tokens_forwarded(self):
        provider, client = self._make_provider("ok")
        provider.generate([], max_new_tokens=256)
        _, kwargs = client.chat.completions.create.call_args
        assert kwargs.get("max_tokens") == 256 or client.chat.completions.create.call_args[1].get("max_tokens") == 256

    def test_extra_kwargs_passed(self):
        provider, client = self._make_provider("ok")
        provider.generate([], max_new_tokens=10, temperature=0.3)
        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs.get("temperature") == 0.3

    def test_default_params_merged(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _make_openai_response("ok")
        provider = OpenAIProvider(
            model="gpt-4o",
            client=client,
            default_params={"temperature": 0.1},
        )
        provider.generate([], max_new_tokens=10)
        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs.get("temperature") == 0.1

    def test_missing_openai_package_raises(self):
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(ImportError, match="openai"):
                OpenAIProvider(model="gpt-4o")


# ---------------------------------------------------------------------------
# Anthropic Provider
# ---------------------------------------------------------------------------


def _make_anthropic_response(text: str, stop_reason: str = "end_turn"):
    block = SimpleNamespace(text=text, type="text")
    return SimpleNamespace(content=[block], stop_reason=stop_reason)


class TestAnthropicProvider:
    def _make_provider(self, text: str, stop_reason: str = "end_turn"):
        client = MagicMock()
        client.messages.create.return_value = _make_anthropic_response(text, stop_reason)
        return AnthropicProvider(model="claude-sonnet-4-6", client=client), client

    def test_generate_returns_text(self):
        provider, _ = self._make_provider("Hello from Claude!")
        result = provider.generate([{"role": "user", "content": "Hi"}], max_new_tokens=100)
        assert result.text == "Hello from Claude!"

    def test_end_turn_maps_to_stop(self):
        provider, _ = self._make_provider("Done.", "end_turn")
        result = provider.generate([], max_new_tokens=10)
        assert result.finish_reason == "stop"

    def test_max_tokens_maps_to_length(self):
        provider, _ = self._make_provider("cut off", "max_tokens")
        result = provider.generate([], max_new_tokens=10)
        assert result.finish_reason == "length"

    def test_system_prompt_included(self):
        client = MagicMock()
        client.messages.create.return_value = _make_anthropic_response("ok")
        provider = AnthropicProvider(
            client=client,
            system_prompt="You are a helpful assistant.",
        )
        provider.generate([], max_new_tokens=10)
        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs.get("system") == "You are a helpful assistant."

    def test_missing_anthropic_package_raises(self):
        with patch.dict("sys.modules", {"anthropic": None}):
            with pytest.raises(ImportError, match="anthropic"):
                AnthropicProvider()


# ---------------------------------------------------------------------------
# HuggingFace Provider
# ---------------------------------------------------------------------------


def _make_hf_response(content: str, finish_reason: str = "stop"):
    choice = SimpleNamespace(
        message=SimpleNamespace(content=content),
        finish_reason=finish_reason,
    )
    return SimpleNamespace(choices=[choice])


class TestHuggingFaceProvider:
    def _make_provider(self, content: str, finish_reason: str = "stop"):
        client = MagicMock()
        client.chat_completion.return_value = _make_hf_response(content, finish_reason)
        return (
            HuggingFaceProvider(model="meta-llama/Llama-3-8B-Instruct", client=client),
            client,
        )

    def test_generate_returns_text(self):
        provider, _ = self._make_provider("Hi there!")
        result = provider.generate([{"role": "user", "content": "Hey"}], max_new_tokens=50)
        assert result.text == "Hi there!"

    def test_eos_token_maps_to_stop(self):
        provider, _ = self._make_provider("Done", "eos_token")
        result = provider.generate([], max_new_tokens=10)
        assert result.finish_reason == "stop"

    def test_length_finish_reason_preserved(self):
        provider, _ = self._make_provider("truncated", "length")
        result = provider.generate([], max_new_tokens=10)
        assert result.finish_reason == "length"

    def test_missing_huggingface_hub_raises(self):
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            with pytest.raises(ImportError, match="huggingface_hub"):
                HuggingFaceProvider(model="some/model")
