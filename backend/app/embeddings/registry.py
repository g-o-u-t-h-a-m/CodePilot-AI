"""Provider registry for managing embedding providers.

This module implements the Registry Pattern to decouple the EmbeddingEngine
from specific provider implementations. New providers can be registered
without modifying the engine or existing providers.

The registry follows the Open/Closed Principle: open for extension
(new providers can be added) but closed for modification (the registry
interface remains stable).
"""

import logging
from typing import Dict, Optional, Type

from app.embeddings.provider import EmbeddingProvider

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """Registry for managing embedding providers.

    The registry maps provider names to provider classes, allowing
    the engine to dynamically select and instantiate providers at runtime.

    Design principles:
    - Single Responsibility: Only manages provider registration and retrieval
    - Open/Closed: New providers can be added without modifying this class
    - Dependency Inversion: Depends on EmbeddingProvider abstraction
    """

    def __init__(self):
        """Initialize the provider registry."""
        self._providers: Dict[str, Type[EmbeddingProvider]] = {}
        self._default_provider: Optional[str] = None
        logger.info("ProviderRegistry initialized")

    def register(
        self,
        name: str,
        provider_class: Type[EmbeddingProvider],
        set_as_default: bool = False
    ) -> None:
        """Register an embedding provider.

        Args:
            name: Unique identifier for the provider
            provider_class: Provider class to register
            set_as_default: If True, set as the default provider

        Raises:
            ValueError: If name is already registered
        """
        if name in self._providers:
            logger.warning(f"Provider '{name}' is already registered. Overwriting.")

        self._providers[name] = provider_class
        logger.info(f"Registered provider: {name} -> {provider_class.__name__}")

        if set_as_default or self._default_provider is None:
            self._default_provider = name
            logger.info(f"Set default provider: {name}")

    def get(self, name: Optional[str] = None) -> Type[EmbeddingProvider]:
        """Retrieve a provider class by name.

        Args:
            name: Provider name. If None, returns the default provider.

        Returns:
            Provider class

        Raises:
            ValueError: If provider is not found or no default is set
        """
        # Use default if no name specified
        if name is None:
            if self._default_provider is None:
                raise ValueError("No default provider set")
            name = self._default_provider

        # Retrieve provider
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
            name: Provider name

        Returns:
            True if registered, False otherwise
        """
        return name in self._providers

    def list_providers(self) -> list[str]:
        """List all registered provider names.

        Returns:
            List of provider names
        """
        return list(self._providers.keys())

    def get_default_provider(self) -> Optional[str]:
        """Get the name of the default provider.

        Returns:
            Default provider name or None if not set
        """
        return self._default_provider

    def set_default(self, name: str) -> None:
        """Set the default provider.

        Args:
            name: Provider name to set as default

        Raises:
            ValueError: If provider is not registered
        """
        if name not in self._providers:
            raise ValueError(f"Provider '{name}' not registered")

        self._default_provider = name
        logger.info(f"Set default provider: {name}")


# Global registry instance
_global_registry = ProviderRegistry()


def get_registry() -> ProviderRegistry:
    """Get the global provider registry instance.

    Returns:
        Global ProviderRegistry instance
    """
    return _global_registry


def register_provider(
    name: str,
    provider_class: Type[EmbeddingProvider],
    set_as_default: bool = False
) -> None:
    """Register a provider in the global registry.

    Args:
        name: Unique identifier for the provider
        provider_class: Provider class to register
        set_as_default: If True, set as the default provider
    """
    _global_registry.register(name, provider_class, set_as_default)
