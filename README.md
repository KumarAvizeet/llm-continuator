# llm-continuator

[![CI](https://github.com/kulkarniadikumar/llm-continuator/actions/workflows/ci.yml/badge.svg)](https://github.com/kulkarniadikumar/llm-continuator/actions)
[![PyPI](https://img.shields.io/pypi/v/llm-continuator)](https://pypi.org/project/llm-continuator/)
[![Python](https://img.shields.io/pypi/pyversions/llm-continuator)](https://pypi.org/project/llm-continuator/)
[![License: KUMAR](https://img.shields.io/badge/License-KUMAR-yellow.svg)](LICENSE)

**Stop losing output to token limits.** `llm-continuator` detects when an LLM response was cut off mid-sentence, mid-code, or mid-list, automatically issues a continuation request, and stitches the parts into one coherent response — across OpenAI, Anthropic, and HuggingFace APIs.

---

## Features

- **Truncation detection** — syntactic checks for open code fences, unbalanced brackets, dangling sentences, and API `finish_reason == "length"` signals
- **Automatic retry** — continuation prompts are crafted to pick up exactly where the model stopped
- **Output stitching** — three strategies (`concat`, `smart`, `code_aware`) to eliminate duplicated tokens at segment joins
- **Configurable** — max retries, token budget, sensitivity level, custom continuation templates
- **Multi-provider** — OpenAI, Anthropic, HuggingFace; bring your own by subclassing `BaseProvider`
- **Zero mandatory dependencies** — provider SDKs are optional extras

---

## Installation

```bash
# Core only (bring your own provider client)
pip install llm-continuator

# With a specific provider
pip install "llm-continuator[openai]"
pip install "llm-continuator[anthropic]"
pip install "llm-continuator[huggingface]"

# All providers
pip install "llm-continuator[all]"

# Development
pip install "llm-continuator[dev]"
```

---

## Quick Start

### OpenAI

```python
from llm_continuator import Continuator, ContinuatorConfig
from llm_continuator.providers import OpenAIProvider

provider = OpenAIProvider(model="gpt-4o")           # api_key from OPENAI_API_KEY
config = ContinuatorConfig(
    max_retries=5,
    max_new_tokens=1024,
    stitching_strategy="smart",
    truncation_sensitivity="medium",
)
cont = Continuator(provider, config)

result = cont.complete([
    {"role": "user", "content": "Write a 1000-word essay on the history of Python."}
])

print(result.text)
print(f"Attempts: {result.attempts}, retried: {result.was_truncated}")
```

### Anthropic

```python
from llm_continuator import Continuator
from llm_continuator.providers import AnthropicProvider

provider = AnthropicProvider(
    model="claude-sonnet-4-6",   # api_key from ANTHROPIC_API_KEY
    system_prompt="You are a helpful technical writer.",
)
cont = Continuator(provider)

result = cont.complete([
    {"role": "user", "content": "Explain every Python data structure in detail."}
])
print(result.text)
```

### HuggingFace

```python
from llm_continuator import Continuator
from llm_continuator.providers import HuggingFaceProvider

provider = HuggingFaceProvider(
    model="meta-llama/Llama-3-8B-Instruct",
    # api_key from HF_TOKEN env var, or pass explicitly
)
cont = Continuator(provider)

result = cont.complete([
    {"role": "user", "content": "Describe the theory of relativity."}
])
print(result.text)
```

---

## Configuration Reference

```python
from llm_continuator import ContinuatorConfig

config = ContinuatorConfig(
    # Max continuation attempts after the initial call
    max_retries=5,

    # Token budget for each individual LLM call
    max_new_tokens=1024,

    # How to join partial outputs: "concat" | "smart" | "code_aware"
    stitching_strategy="smart",

    # How aggressively to detect truncation: "low" | "medium" | "high"
    # low    → only trust finish_reason == "length"
    # medium → also check code fences, unbalanced brackets, sentence endings
    # high   → additionally catch mid-list, dangling bullets, etc.
    truncation_sensitivity="medium",

    # Template for the continuation message; {partial} is the last N chars
    continuation_prompt_template=(
        "Continue exactly from where you stopped. "
        "Do not repeat any text. "
        "Here is the end of your previous response:\n\n...{partial}"
    ),

    # How many trailing characters from accumulated output to include in
    # the continuation prompt for context
    continuation_window=300,

    # If True, raises MaxRetriesExceededError when retries are exhausted.
    # If False, returns whatever has been accumulated.
    raise_on_max_retries=True,
)
```

---

## Stitching Strategies

| Strategy | Description | Best for |
|---|---|---|
| `concat` | Simple concatenation, no overlap removal | Speed, when you know the model won't repeat text |
| `smart` | Removes overlapping tokens at segment boundaries | General prose |
| `code_aware` | Like `smart` but also repairs unclosed code fences | Responses containing code blocks |

---

## Custom Provider

Subclass `BaseProvider` to wrap any LLM API:

```python
from llm_continuator.providers.base import BaseProvider, GenerationResult

class MyProvider(BaseProvider):
    def generate(self, messages, max_new_tokens, **kwargs):
        # Call your LLM here
        raw = my_llm_client.generate(messages, max_tokens=max_new_tokens)
        return GenerationResult(
            text=raw.text,
            finish_reason=raw.stop_reason,   # "stop" | "length" | None
            raw=raw,                          # optional, for introspection
        )
```

---

## Error Handling

```python
from llm_continuator import Continuator, ContinuatorConfig, MaxRetriesExceededError

cfg = ContinuatorConfig(max_retries=3, raise_on_max_retries=True)
cont = Continuator(provider, cfg)

try:
    result = cont.complete(messages)
except MaxRetriesExceededError as e:
    print(f"Gave up after {e.attempts} attempts")
    print(f"Partial output: {e.partial_output[:200]}")
```

---

## CompletionResult Fields

| Field | Type | Description |
|---|---|---|
| `text` | `str` | The fully assembled response |
| `attempts` | `int` | Total LLM calls made (1 = no retry needed) |
| `segments` | `list[str]` | Raw text from each generation call |
| `was_truncated` | `bool` | Whether any retry was triggered |
| `last_finish_reason` | `str \| None` | `finish_reason` of the final segment |

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

Test matrix covers Python 3.9–3.12. All tests use mock provider clients — no real API keys required.

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Run tests: `pytest`
4. Lint: `ruff check llm_continuator tests`
5. Open a pull request

---

## License

KUMAR — see [LICENSE](LICENSE).
