"""Unit tests for the Continuator core — uses mock providers."""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from llm_continuator import (
    Continuator,
    ContinuatorConfig,
    MaxRetriesExceededError,
)
from llm_continuator.providers.base import BaseProvider, GenerationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_result(text: str, finish_reason: str = "stop") -> GenerationResult:
    return GenerationResult(text=text, finish_reason=finish_reason)


class ScriptedProvider(BaseProvider):
    """Provider that replays a predefined sequence of GenerationResults."""

    def __init__(self, results: List[GenerationResult]) -> None:
        self._results = iter(results)

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int,
        **kwargs: Any,
    ) -> GenerationResult:
        try:
            return next(self._results)
        except StopIteration:
            raise RuntimeError("ScriptedProvider: no more scripted results")


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


class TestContinuatorHappyPath:
    def test_single_complete_response(self):
        provider = ScriptedProvider([make_result("Hello, world.", "stop")])
        cont = Continuator(provider)
        result = cont.complete([{"role": "user", "content": "Hi"}])

        assert result.text == "Hello, world."
        assert result.attempts == 1
        assert result.was_truncated is False
        assert result.last_finish_reason == "stop"

    def test_completes_after_one_retry(self):
        provider = ScriptedProvider(
            [
                make_result("This is a very long resp", "length"),
                make_result("onse. Done.", "stop"),
            ]
        )
        cont = Continuator(provider, ContinuatorConfig(max_retries=3))
        result = cont.complete([{"role": "user", "content": "Tell me something"}])

        assert result.attempts == 2
        assert result.was_truncated is True
        # Smart stitching should join the overlap correctly.
        assert "Done." in result.text

    def test_completes_after_two_retries(self):
        provider = ScriptedProvider(
            [
                make_result("Part one", "length"),
                make_result(", part two", "length"),
                make_result(", part three.", "stop"),
            ]
        )
        cont = Continuator(provider, ContinuatorConfig(max_retries=5))
        result = cont.complete([{"role": "user", "content": "Go"}])

        assert result.attempts == 3
        assert "Part one" in result.text
        assert "part three." in result.text

    def test_segments_list_populated(self):
        provider = ScriptedProvider(
            [
                make_result("Segment A. ", "length"),
                make_result("Segment B.", "stop"),
            ]
        )
        cont = Continuator(provider)
        result = cont.complete([{"role": "user", "content": "x"}])

        assert len(result.segments) == 2
        assert result.segments[0] == "Segment A. "
        assert result.segments[1] == "Segment B."


# ---------------------------------------------------------------------------
# Retry-exhaustion tests
# ---------------------------------------------------------------------------


class TestMaxRetriesExhausted:
    def test_raises_by_default(self):
        # Always truncated provider.
        provider = ScriptedProvider(
            [make_result("still going", "length")] * 10
        )
        cfg = ContinuatorConfig(max_retries=2, raise_on_max_retries=True)
        cont = Continuator(provider, cfg)

        with pytest.raises(MaxRetriesExceededError) as exc_info:
            cont.complete([{"role": "user", "content": "x"}])

        assert exc_info.value.attempts == 3  # initial + 2 retries

    def test_returns_partial_when_not_raising(self):
        provider = ScriptedProvider(
            [make_result("still going", "length")] * 10
        )
        cfg = ContinuatorConfig(max_retries=2, raise_on_max_retries=False)
        cont = Continuator(provider, cfg)

        result = cont.complete([{"role": "user", "content": "x"}])
        assert "still going" in result.text

    def test_max_retries_exceeded_error_has_partial(self):
        provider = ScriptedProvider(
            [make_result("partial content", "length")] * 10
        )
        cfg = ContinuatorConfig(max_retries=1, raise_on_max_retries=True)
        cont = Continuator(provider, cfg)

        with pytest.raises(MaxRetriesExceededError) as exc_info:
            cont.complete([{"role": "user", "content": "x"}])

        assert "partial content" in exc_info.value.partial_output


# ---------------------------------------------------------------------------
# Continuation prompt tests
# ---------------------------------------------------------------------------


class TestContinuationPromptBuilding:
    def test_continuation_message_is_appended(self):
        """Verify the continuation prompt contains the accumulated output."""
        captured_messages: List[List[Dict]] = []

        class CapturingProvider(BaseProvider):
            call_count = 0

            def generate(self, messages, max_new_tokens, **kwargs):
                captured_messages.append(messages)
                self.call_count += 1
                if self.call_count == 1:
                    return make_result("Partial response", "length")
                return make_result(" complete.", "stop")

        provider = CapturingProvider()
        cont = Continuator(provider)
        cont.complete([{"role": "user", "content": "Tell me a story"}])

        # Second call should include the partial assistant turn.
        second_call_msgs = captured_messages[1]
        roles = [m["role"] for m in second_call_msgs]
        assert "assistant" in roles
        assert "user" in roles
        # The continuation prompt should reference the partial text.
        continuation_user = next(
            m for m in reversed(second_call_msgs) if m["role"] == "user"
        )
        assert "Partial response" in continuation_user["content"]

    def test_custom_continuation_template(self):
        """Custom template is interpolated correctly."""
        captured: List[str] = []

        class CapturingProvider(BaseProvider):
            call_count = 0

            def generate(self, messages, max_new_tokens, **kwargs):
                self.call_count += 1
                if self.call_count == 1:
                    return make_result("Hello", "length")
                captured.append(messages[-1]["content"])
                return make_result(" World.", "stop")

        provider = CapturingProvider()
        cfg = ContinuatorConfig(
            continuation_prompt_template="CONTINUE: [{partial}]",
            continuation_window=50,
        )
        cont = Continuator(provider, cfg)
        cont.complete([{"role": "user", "content": "x"}])

        assert captured and "CONTINUE:" in captured[0]
        assert "Hello" in captured[0]


# ---------------------------------------------------------------------------
# Stitching strategy integration
# ---------------------------------------------------------------------------


class TestStitchingIntegration:
    @pytest.mark.parametrize("strategy", ["concat", "smart", "code_aware"])
    def test_all_strategies_produce_string(self, strategy):
        provider = ScriptedProvider(
            [
                make_result("Start ", "length"),
                make_result("End.", "stop"),
            ]
        )
        cfg = ContinuatorConfig(stitching_strategy=strategy)
        cont = Continuator(provider, cfg)
        result = cont.complete([{"role": "user", "content": "x"}])
        assert isinstance(result.text, str)
        assert len(result.text) > 0


# ---------------------------------------------------------------------------
# Provider interaction — max_new_tokens forwarding
# ---------------------------------------------------------------------------


class TestMaxNewTokensForwarding:
    def test_max_new_tokens_passed_to_provider(self):
        recorded: List[int] = []

        class RecordingProvider(BaseProvider):
            def generate(self, messages, max_new_tokens, **kwargs):
                recorded.append(max_new_tokens)
                return make_result("Done.", "stop")

        provider = RecordingProvider()
        cfg = ContinuatorConfig(max_new_tokens=512)
        cont = Continuator(provider, cfg)
        cont.complete([{"role": "user", "content": "x"}])

        assert recorded == [512]

    def test_extra_kwargs_forwarded(self):
        recorded_kwargs: List[Dict] = []

        class RecordingProvider(BaseProvider):
            def generate(self, messages, max_new_tokens, **kwargs):
                recorded_kwargs.append(kwargs)
                return make_result("Done.", "stop")

        provider = RecordingProvider()
        cont = Continuator(provider)
        cont.complete([{"role": "user", "content": "x"}], temperature=0.5, top_p=0.9)

        assert recorded_kwargs[0] == {"temperature": 0.5, "top_p": 0.9}
