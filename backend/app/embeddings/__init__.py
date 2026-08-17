"""Embeddings module for CodePilot AI.

This module provides a flexible, extensible architecture for generating
embeddings from code chunks using the Provider Pattern and Registry Pattern.

Main components:
    - EmbeddingEngine: Main entry point for embedding operations
    - EmbeddingProvider: Abstract base class for providers
    - ProviderRegistry: Registry for managing providers
    - EmbeddingCache: Cache for avoiding redundant embeddings
    - EmbeddingRecord: Data model for embedding records

Design patterns:
    - Strategy Pattern: Different providers implement the same interface
    - Registry Pattern: Providers are registered and retrieved dynamically
    - Dependency Inversion: Engine depends on abstractions, not implementations

Example usage:
    from app.embeddings import (
        EmbeddingEngine,
        initialize_providers,
    )
    from app.chunking.models import CodeChunk

    # Initialize the provider registry
    initialize_providers()

    # Create the embedding engine
    engine = EmbeddingEngine()

    # Embed a code chunk
    record = engine.embed(chunk)

    # Access embedding data
    print(f"Embedding dimension: {record.dimension}")
    print(f"Model used: {record.model_name}")

Adding new providers:
    To add a new embedding provider (e.g., OpenAI, Nomic, Jina):

    1. Create a new provider class that inherits from EmbeddingProvider
    2. Implement the embed(), get_model_name(), and get_dimension() methods
    3. Register the provider in initialize_providers()

    Example:
        from app.embeddings import EmbeddingProvider, register_provider

        class OpenAIProvider(EmbeddingProvider):
            def embed(self, chunk):
                # Implementation
                pass

            def get_model_name(self):
                return "text-embedding-3-small"

            def get_dimension(self):
                return 1536

        # Register the provider
        register_provider("openai", OpenAIProvider)

    The engine can then use the new provider without any modifications.
"""

import logging

from app.embeddings.cache import EmbeddingCache, PersistentEmbeddingCache
from app.embeddings.engine import EmbeddingEngine
from app.embeddings.models import EmbeddingRecord
from app.embeddings.provider import EmbeddingProvider
from app.embeddings.providers import BGEProvider
from app.embeddings.registry import (
    get_registry,
    register_provider,
)

logger = logging.getLogger(__name__)


def initialize_providers():
    """Initialize and register all embedding providers.

    This function must be called once during application startup
    to populate the provider registry.

    The registry maps provider names to provider classes. This approach
    follows the Open/Closed Principle: new providers can be added
    without modifying the EmbeddingEngine.

    Currently registered providers:
        - bge: BAAI/bge-small-en-v1.5 (default)

    To add additional providers, register them here:
        register_provider("openai", OpenAIProvider)
        register_provider("nomic", NomicProvider)
        register_provider("jina", JinaProvider)
    """
    logger.info("Initializing embedding providers")

    # Register BGE provider as default
    register_provider(
        name="bge",
        provider_class=BGEProvider,
        set_as_default=True
    )

    logger.info("Embedding providers initialized successfully")


__all__ = [
    "EmbeddingEngine",
    "EmbeddingRecord",
    "EmbeddingProvider",
    "EmbeddingCache",
    "PersistentEmbeddingCache",
    "BGEProvider",
    "initialize_providers",
    "get_registry",
    "register_provider",
]
