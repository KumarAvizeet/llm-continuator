"""Unit tests for ContinuatorConfig validation."""

import pytest
from llm_continuator.config import ContinuatorConfig
from llm_continuator.exceptions import ConfigurationError


class TestContinuatorConfig:
    def test_defaults_are_valid(self):
        cfg = ContinuatorConfig()
        assert cfg.max_retries == 5
        assert cfg.max_new_tokens == 1024
        assert cfg.stitching_strategy == "smart"
        assert cfg.truncation_sensitivity == "medium"
        assert cfg.raise_on_max_retries is True

    def test_custom_values(self):
        cfg = ContinuatorConfig(
            max_retries=3,
            max_new_tokens=512,
            stitching_strategy="code_aware",
            truncation_sensitivity="high",
            raise_on_max_retries=False,
        )
        assert cfg.max_retries == 3
        assert cfg.stitching_strategy == "code_aware"

    def test_invalid_max_retries(self):
        with pytest.raises(ConfigurationError, match="max_retries"):
            ContinuatorConfig(max_retries=0)

    def test_invalid_max_new_tokens(self):
        with pytest.raises(ConfigurationError, match="max_new_tokens"):
            ContinuatorConfig(max_new_tokens=0)

    def test_invalid_continuation_window(self):
        with pytest.raises(ConfigurationError, match="continuation_window"):
            ContinuatorConfig(continuation_window=5)

    def test_invalid_stitching_strategy(self):
        with pytest.raises(ConfigurationError, match="stitching_strategy"):
            ContinuatorConfig(stitching_strategy="magic")

    def test_invalid_truncation_sensitivity(self):
        with pytest.raises(ConfigurationError, match="truncation_sensitivity"):
            ContinuatorConfig(truncation_sensitivity="ultra")
