"""Unit tests for truncation detectors."""

import pytest
from llm_continuator.detectors import (
    TruncationDetector,
    _ends_mid_list,
    _ends_mid_sentence,
    _open_brackets,
    _open_code_fence,
)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestOpenCodeFence:
    def test_no_fence(self):
        assert _open_code_fence("Hello world.") is False

    def test_closed_fence(self):
        text = "Here:\n```python\nx = 1\n```\nDone."
        assert _open_code_fence(text) is False

    def test_open_fence(self):
        text = "Here:\n```python\nx = 1\n"
        assert _open_code_fence(text) is True

    def test_multiple_closed_fences(self):
        text = "```\ncode1\n```\n```\ncode2\n```"
        assert _open_code_fence(text) is False

    def test_odd_number_of_fences(self):
        text = "```\ncode1\n```\n```\ncode2"
        assert _open_code_fence(text) is True


class TestOpenBrackets:
    def test_balanced(self):
        assert _open_brackets("f(x, [1, 2])") is False

    def test_unbalanced_paren(self):
        assert _open_brackets("f(x, [1, 2]") is True

    def test_unbalanced_bracket(self):
        assert _open_brackets("arr[0") is True

    def test_empty_string(self):
        assert _open_brackets("") is False

    def test_string_with_brackets_inside_quotes(self):
        # Brackets inside strings should not count.
        assert _open_brackets('"hello (world"') is False


class TestEndsMidSentence:
    def test_proper_period(self):
        assert _ends_mid_sentence("This is done.") is False

    def test_question_mark(self):
        assert _ends_mid_sentence("Really?") is False

    def test_trailing_word(self):
        assert _ends_mid_sentence("This is truncat") is True

    def test_empty(self):
        assert _ends_mid_sentence("") is False

    def test_ends_with_comma(self):
        assert _ends_mid_sentence("First item,") is True

    def test_ends_with_closing_paren(self):
        assert _ends_mid_sentence("(see above)") is False


class TestEndsMidList:
    def test_trailing_comma(self):
        assert _ends_mid_list("item 1,") is True

    def test_dangling_bullet(self):
        assert _ends_mid_list("- ") is True

    def test_dangling_numbered(self):
        assert _ends_mid_list("1. ") is True

    def test_complete_list(self):
        assert _ends_mid_list("- item one\n- item two") is False


# ---------------------------------------------------------------------------
# TruncationDetector tests
# ---------------------------------------------------------------------------


class TestTruncationDetectorLow:
    def setup_method(self):
        self.det = TruncationDetector(sensitivity="low")

    def test_length_finish_reason(self):
        assert self.det.is_truncated("some text", finish_reason="length") is True

    def test_max_tokens_finish_reason(self):
        assert self.det.is_truncated("some text", finish_reason="max_tokens") is True

    def test_stop_finish_reason_complete_text(self):
        assert self.det.is_truncated("Done.", finish_reason="stop") is False

    def test_stop_finish_reason_open_fence(self):
        # Low sensitivity ignores syntactic signals.
        assert self.det.is_truncated("```\ncode", finish_reason="stop") is False

    def test_none_finish_reason(self):
        assert self.det.is_truncated("truncated", finish_reason=None) is False


class TestTruncationDetectorMedium:
    def setup_method(self):
        self.det = TruncationDetector(sensitivity="medium")

    def test_length_finish_reason(self):
        assert self.det.is_truncated("anything", finish_reason="length") is True

    def test_open_code_fence(self):
        assert self.det.is_truncated("```\ncode\n", finish_reason="stop") is True

    def test_closed_code_fence(self):
        assert self.det.is_truncated("```\ncode\n```", finish_reason="stop") is False

    def test_truncated_sentence(self):
        assert self.det.is_truncated("The answer is clearly", finish_reason="stop") is True

    def test_complete_sentence(self):
        assert self.det.is_truncated("The answer is 42.", finish_reason="stop") is False

    def test_unbalanced_paren(self):
        assert self.det.is_truncated("result = f(x", finish_reason="stop") is True


class TestTruncationDetectorHigh:
    def setup_method(self):
        self.det = TruncationDetector(sensitivity="high")

    def test_trailing_comma_in_list(self):
        text = "Items:\n- alpha,\n- beta,"
        assert self.det.is_truncated(text, finish_reason="stop") is True

    def test_dangling_bullet(self):
        assert self.det.is_truncated("- ", finish_reason="stop") is True

    def test_complete_response(self):
        text = "Summary:\n- Point A\n- Point B\n\nThat covers everything."
        assert self.det.is_truncated(text, finish_reason="stop") is False
