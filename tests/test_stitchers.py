"""Unit tests for output stitching strategies."""

import pytest
from llm_continuator.stitchers import (
    code_aware_stitch,
    concat_stitch,
    smart_stitch,
    stitch,
)


class TestConcatStitch:
    def test_empty(self):
        assert concat_stitch([]) == ""

    def test_single_segment(self):
        assert concat_stitch(["hello"]) == "hello"

    def test_multiple_segments(self):
        assert concat_stitch(["foo", "bar", "baz"]) == "foobarbaz"

    def test_preserves_whitespace(self):
        assert concat_stitch(["hello ", "world"]) == "hello world"


class TestSmartStitch:
    def test_empty(self):
        assert smart_stitch([]) == ""

    def test_single_segment(self):
        assert smart_stitch(["hello world."]) == "hello world."

    def test_no_overlap(self):
        result = smart_stitch(["Part one. ", "Part two."])
        assert result == "Part one. Part two."

    def test_overlap_removed(self):
        # Simulate model repeating context from the continuation prompt.
        a = "The quick brown fox"
        b = "brown fox jumps over"
        result = smart_stitch([a, b])
        assert result == "The quick brown fox jumps over"

    def test_exact_match_overlap(self):
        a = "Hello world"
        b = "world!"
        result = smart_stitch([a, b])
        assert result == "Hello world!"

    def test_three_segments_overlap(self):
        segs = ["Step one, step two,", " two, step three,", " three, done."]
        result = smart_stitch(segs)
        assert "step two" in result
        assert "step three" in result
        assert result.endswith("done.")


class TestCodeAwareStitch:
    def test_closes_open_fence(self):
        a = "Here is code:\n```python\ndef foo():\n    return 1\n"
        b = "    return 2\n```\nEnd."
        result = code_aware_stitch([a, b])
        # The fence should be properly closed somewhere in the result.
        assert result.count("```") % 2 == 0

    def test_no_fence_acts_like_smart(self):
        a = "The quick brown"
        b = "brown fox."
        assert code_aware_stitch([a, b]) == smart_stitch([a, b])

    def test_single_segment(self):
        assert code_aware_stitch(["hello"]) == "hello"

    def test_empty(self):
        assert code_aware_stitch([]) == ""


class TestStitchDispatch:
    def test_concat(self):
        assert stitch(["a", "b"], "concat") == "ab"

    def test_smart(self):
        assert stitch(["hello world", "world!"], "smart") == "hello world!"

    def test_code_aware(self):
        result = stitch(["```\ncode", " more code\n```"], "code_aware")
        assert isinstance(result, str)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown stitching strategy"):
            stitch(["a"], "nonexistent")
