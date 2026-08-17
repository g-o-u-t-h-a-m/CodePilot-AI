"""LLM provider implementations.

This package contains concrete implementations of LLM providers. Each
provider implements the LLMProvider interface and can be registered with the
LLM provider registry.

Available providers:
    - MockLLMProvider: Deterministic mock for 0-cost local testing
    - OpenAICompatibleProvider: Configurable OpenAI-compatible HTTP provider
"""

from app.llm.providers.mock import MockLLMProvider
from app.llm.providers.openai_compatible import OpenAICompatibleProvider

__all__ = [
    "MockLLMProvider",
    "OpenAICompatibleProvider",
]