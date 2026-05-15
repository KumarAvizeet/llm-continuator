"""llm-continuator — Reliable complete responses from any LLM API.

Prevents truncated/incomplete outputs by detecting premature cuts and
issuing continuation requests, then stitching the parts into one response.

Quick start
-----------
>>> from llm_continuator import Continuator, ContinuatorConfig
>>> from llm_continuator.providers import OpenAIProvider
>>>
>>> provider = OpenAIProvider(model="gpt-4o")
>>> cont = Continuator(provider)
>>> result = cont.complete([{"role": "user", "content": "Write a long essay."}])
>>> print(result.text)
"""

from .config import ContinuatorConfig
from .core import CompletionResult, Continuator
from .exceptions import (
    ConfigurationError,
    LLMContinuatorError,
    MaxRetriesExceededError,
    ProviderError,
    StitchingError,
)
from .providers import (
    AnthropicProvider,
    BaseProvider,
    GenerationResult,
    HuggingFaceProvider,
    OpenAIProvider,
)

__version__ = "0.1.0"
__all__ = [
    # Core
    "Continuator",
    "CompletionResult",
    "ContinuatorConfig",
    # Providers
    "BaseProvider",
    "GenerationResult",
    "OpenAIProvider",
    "AnthropicProvider",
    "HuggingFaceProvider",
    # Exceptions
    "LLMContinuatorError",
    "MaxRetriesExceededError",
    "ProviderError",
    "StitchingError",
    "ConfigurationError",
]
