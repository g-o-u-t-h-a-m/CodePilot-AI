"""Provider registry for managing LLM providers.

This module implements the Registry Pattern (mirroring
app.embeddings.registry) to decouple the RAG service from specific LLM
provider implementations. New providers can be registered without modifying
the service or existing providers.

The registry follows the Open/Closed Principle: open for extension (new
providers can be added) but closed for modification (the registry interface
remains stable).
"""

import logging
from typing import Dict, Optional, Type

from app.llm.provider import LLMProvider

logger = logging.getLogger(__name__)


class LLMRegistry:
    """Registry for managing LLM providers.

    The registry maps provider names to provider classes, allowing the
    application to dynamically select and instantiate providers at runtime
    (for example LLM_PROVIDER=mock).

    Design principles:
    - Single Responsibility: Only manages provider registration and retrieval
    - Open/Closed: New providers can be added without modifying this class
    - Dependency Inversion: Depends on the LLMProvider abstraction
    """

    def __init__(self):
        """Initialize the provider registry."""
        self._providers: Dict[str, Type[LLMProvider]] = {}
        self._default_provider: Optional[str] = None
        logger.info("LLMRegistry initialized")

    def register(
        self,
        name: str,
        provider_class: Type[LLMProvider],
        set_as_default: bool = False,
    ) -> None:
        """Register an LLM provider.

        Args:
            name: Unique identifier for the provider.
            provider_class: Provider class to register.
            set_as_default: If True, set as the default provider.

        Raises:
            ValueError: If name is not a non-empty string or the class is
                not an LLMProvider subclass.
        """
        if not name or not name.strip():
            raise ValueError("Provider name must be a non-empty string")
        if not issubclass(provider_class, LLMProvider):
            raise ValueError(
                f"provider_class must subclass LLMProvider, "
                f"got {provider_class.__name__}"
            )

        if name in self._providers:
            logger.warning(
                f"Provider '{name}' is already registered. Overwriting."
            )

        self._providers[name] = provider_class
        logger.info(f"Registered provider: {name} -> {provider_class.__name__}")

        if set_as_default or self._default_provider is None:
            self._default_provider = name
            logger.info(f"Set default provider: {name}")

    def get(self, name: Optional[str] = None) -> Type[LLMProvider]:
        """Retrieve a provider class by name.

        Args:
            name: Provider name. If None, returns the default provider.

        Returns:
            Provider class.

        Raises:
            ValueError: If provider is not found or no default is set.
        """
        if name is None:
            if self._default_provider is None:
                raise ValueError("No default provider set")
            name = self._default_provider

        if name not in self._providers:
            available = list(self._providers.keys())
            raise ValueError(
                f"Provider '{name}' not found. "
                f"Available providers: {available}"
            )

        logger.debug(f"Retrieved provider: {name}")
        return self._providers[name]

    def has(self, name: str) -> bool:
        """Check if a provider is registered.

        Args:
            name: Provider name.

        Returns:
            True if registered, False otherwise.
        """
        return name in self._providers

    def list_providers(self) -> list[str]:
        """List all registered provider names.

        Returns:
            List of provider names.
        """
        return list(self._providers.keys())

    def get_default_provider(self) -> Optional[str]:
        """Get the name of the default provider.

        Returns:
            Default provider name or None if not set.
        """
        return self._default_provider

    def set_default(self, name: str) -> None:
        """Set the default provider.

        Args:
            name: Provider name to set as default.

        Raises:
            ValueError: If provider is not registered.
        """
        if name not in self._providers:
            raise ValueError(f"Provider '{name}' not registered")

        self._default_provider = name
        logger.info(f"Set default provider: {name}")


# Global registry instance
_global_registry = LLMRegistry()


def get_registry() -> LLMRegistry:
    """Get the global LLM provider registry instance.

    Returns:
        Global LLMRegistry instance.
    """
    return _global_registry


def register_provider(
    name: str,
    provider_class: Type[LLMProvider],
    set_as_default: bool = False,
) -> None:
    """Register a provider in the global registry.

    Args:
        name: Unique identifier for the provider.
        provider_class: Provider class to register.
        set_as_default: If True, set as the default provider.
    """
    _global_registry.register(name, provider_class, set_as_default)


__all__ = ["LLMRegistry", "get_registry", "register_provider"]