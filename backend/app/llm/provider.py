"""Abstract base class for LLM providers.

This module defines the interface that all LLM providers must implement.
Using the Strategy Pattern (mirroring app.embeddings.provider) allows
different LLM providers to be used interchangeably without modifying the
RAG service.

Implementations must NOT assume a specific model; model name, base URL, and
API key are read from configuration/environment by each provider. No provider
may hard-code API keys, tokens, or secrets.
"""

from abc import ABC, abstractmethod

from app.llm.models import LLMResponse


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    An LLM provider is responsible for:
    1. Sending a prompt to an LLM backend
    2. Returning a normalized LLMResponse
    3. Reading its configuration (base URL, model, API key) from
       configuration/environment rather than hard-coding it

    Implementations should:
    - Keep configuration out of hard-coded defaults where secrets are involved
    - Never commit API keys or tokens
    - Raise meaningful exceptions instead of swallowing failures
    """

    @abstractmethod
    def generate(self, prompt: str) -> LLMResponse:
        """Generate a response for a prompt.

        Args:
            prompt: The prompt to send to the LLM. For the RAG pipeline this
                is the prompt produced by the PromptBuilder.

        Returns:
            LLMResponse containing the generated content and metadata.

        Raises:
            RuntimeError: If generation fails (network, auth, provider error).
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the provider's name (registry key).

        Returns:
            The provider name used in the registry.
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the name of the model this provider is configured to use.

        Returns:
            Model name identifier.
        """
        pass


__all__ = ["LLMProvider"]