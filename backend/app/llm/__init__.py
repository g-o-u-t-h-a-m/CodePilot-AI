"""LLM generation module for CodePilot AI.

Sprint 8 adds the generation half of the RAG pipeline: turning a prompt built
by the PromptBuilder into a grounded natural-language answer using a
pluggable LLM provider.

Main components:
    - LLMProvider: Abstract base class for providers
    - LLMRegistry: Registry for managing providers
    - LLMResponse: Normalized generation result
    - MockLLMProvider: Deterministic mock for 0-cost local testing
    - OpenAICompatibleProvider: Configurable OpenAI-compatible HTTP provider

Design patterns:
    - Strategy Pattern: Different providers implement the same interface
    - Registry Pattern: Providers are registered and retrieved dynamically
    - Dependency Inversion: The RAG service depends on LLMProvider, not a
      concrete implementation

The active provider is selected by configuration (LLM_PROVIDER). No API key
is required when using the mock provider, so the project stays usable with a
0 budget.

Example usage:
    from app.llm import LLMProvider, LLMResponse, initialize_providers, get_registry

    initialize_providers()
    provider_class = get_registry().get()          # or get("mock")
    provider = provider_class()

    response: LLMResponse = provider.generate(built_prompt.prompt)
    print(response.content, response.model_name, response.provider_name)

Adding new providers:
    To add a new LLM provider (e.g. Anthropic, Gemini, a local server):

    1. Create a provider class that inherits from LLMProvider
    2. Implement generate(), get_provider_name(), get_model_name()
    3. Register the provider in initialize_providers()

    Example:
        from app.llm import LLMProvider, register_provider

        class MyProvider(LLMProvider):
            def generate(self, prompt):
                ...

            def get_provider_name(self):
                return "my_provider"

            def get_model_name(self):
                return "my-model"

        register_provider("my_provider", MyProvider)
"""

import logging
import os

from app.llm.models import LLMResponse
from app.llm.provider import LLMProvider
from app.llm.providers import MockLLMProvider, OpenAICompatibleProvider
from app.llm.registry import (
    get_registry,
    register_provider,
)

logger = logging.getLogger(__name__)


def initialize_providers() -> None:
    """Initialize and register all LLM providers.

    This function must be called once during application startup to populate
    the LLM provider registry. The mock provider is registered as the
    default so the application works out of the box with no API key.

    Currently registered providers:
        - mock: MockLLMProvider (default, 0-cost local testing)
        - openai_compatible: OpenAICompatibleProvider (configurable via
          LLM_BASE_URL / LLM_API_KEY / LLM_MODEL)
    """
    logger.info("Initializing LLM providers")

    register_provider(
        name="mock",
        provider_class=MockLLMProvider,
        set_as_default=True,
    )
    register_provider(
        name="openai_compatible",
        provider_class=OpenAICompatibleProvider,
    )

    configured = os.getenv("LLM_PROVIDER")
    if configured and configured not in ("mock", "openai_compatible"):
        logger.warning(
            f"LLM_PROVIDER={configured!r} is not a registered provider; "
            "using default 'mock'."
        )

    logger.info("LLM providers initialized successfully")


def create_llm_provider(provider_name: str | None = None) -> LLMProvider:
    """Create a configured LLM provider instance.

    Reads the LLM_PROVIDER environment variable when no name is given,
    then looks the provider class up in the registry and instantiates it.
    This keeps provider selection entirely config-driven.

    Args:
        provider_name: Provider name. If None, uses LLM_PROVIDER env or the
            registry default ('mock').

    Returns:
        An initialized LLMProvider instance.

    Raises:
        ValueError: If the provider is not registered.
    """
    name = provider_name or os.getenv("LLM_PROVIDER")
    provider_class = get_registry().get(name)
    provider = provider_class()
    logger.info(
        f"Created LLM provider '{provider.get_provider_name()}' "
        f"(model: {provider.get_model_name()})"
    )
    return provider


__all__ = [
    "LLMProvider",
    "LLMResponse",
    "MockLLMProvider",
    "OpenAICompatibleProvider",
    "initialize_providers",
    "create_llm_provider",
    "get_registry",
    "register_provider",
]