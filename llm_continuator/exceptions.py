"""Custom exceptions for llm-continuator."""


class LLMContinuatorError(Exception):
    """Base exception for llm-continuator."""


class MaxRetriesExceededError(LLMContinuatorError):
    """Raised when all retry attempts are exhausted without a complete response."""

    def __init__(self, attempts: int, partial_output: str = ""):
        self.attempts = attempts
        self.partial_output = partial_output
        super().__init__(
            f"LLM output still truncated after {attempts} retry attempts. "
            f"Partial output length: {len(partial_output)} chars."
        )


class ProviderError(LLMContinuatorError):
    """Raised when the underlying LLM provider returns an error."""


class StitchingError(LLMContinuatorError):
    """Raised when output segments cannot be coherently joined."""


class ConfigurationError(LLMContinuatorError):
    """Raised for invalid configuration values."""
