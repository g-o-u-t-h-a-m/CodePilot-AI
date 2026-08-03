"""Strategy registry for managing chunking strategies.

This module implements the Registry Pattern to decouple the ChunkEngine
from specific strategy implementations. New strategies can be registered
without modifying the engine or existing strategies.
"""

import logging
from typing import Dict, Type, Optional

from app.chunking.strategy import ChunkStrategy

logger = logging.getLogger(__name__)


class StrategyRegistry:
    """Registry for managing chunking strategies.

    The registry maps file extensions and languages to appropriate
    chunking strategies following the Open/Closed Principle.
    """

    def __init__(self):
        """Initialize the strategy registry."""
        self._strategies: Dict[str, Type[ChunkStrategy]] = {}
        self._fallback_strategy: Optional[Type[ChunkStrategy]] = None

    def register(self, key: str, strategy: Type[ChunkStrategy]) -> None:
        """Register a strategy for a specific key.

        Args:
            key: Language or extension identifier (e.g., 'python', '.py')
            strategy: Strategy class to register
        """
        self._strategies[key.lower()] = strategy
        logger.debug(f"Registered strategy {strategy.__name__} for key '{key}'")

    def register_fallback(self, strategy: Type[ChunkStrategy]) -> None:
        """Register a fallback strategy for unknown file types.

        Args:
            strategy: Fallback strategy class
        """
        self._fallback_strategy = strategy
        logger.debug(f"Registered fallback strategy: {strategy.__name__}")

    def get_strategy(self, key: str) -> Type[ChunkStrategy]:
        """Retrieve a strategy for the given key.

        Args:
            key: Language or extension identifier

        Returns:
            Strategy class for the key, or fallback strategy if not found
        """
        key_lower = key.lower()

        if key_lower in self._strategies:
            logger.debug(f"Found strategy for key '{key}'")
            return self._strategies[key_lower]

        if self._fallback_strategy:
            logger.debug(f"Using fallback strategy for key '{key}'")
            return self._fallback_strategy

        raise ValueError(f"No strategy registered for '{key}' and no fallback available")

    def list_registered(self) -> list:
        """List all registered strategies.

        Returns:
            List of registered keys
        """
        return list(self._strategies.keys())


# Global registry instance
_global_registry = StrategyRegistry()


def get_registry() -> StrategyRegistry:
    """Get the global strategy registry instance.

    Returns:
        Global StrategyRegistry instance
    """
    return _global_registry


def register_strategy(key: str, strategy: Type[ChunkStrategy]) -> None:
    """Register a strategy in the global registry.

    Args:
        key: Language or extension identifier
        strategy: Strategy class to register
    """
    _global_registry.register(key, strategy)


def register_fallback_strategy(strategy: Type[ChunkStrategy]) -> None:
    """Register a fallback strategy in the global registry.

    Args:
        strategy: Fallback strategy class
    """
    _global_registry.register_fallback(strategy)
