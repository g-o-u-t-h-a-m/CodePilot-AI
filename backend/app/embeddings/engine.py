"""Core embedding engine for generating embeddings from code chunks.

The EmbeddingEngine is the main entry point for embedding operations.
It orchestrates the interaction between providers, cache, and chunks
without knowing implementation details of any component.

The engine follows the Dependency Inversion Principle: it depends on
abstractions (EmbeddingProvider, EmbeddingCache) rather than concrete
implementations.
"""

import logging
from typing import List, Optional

from app.chunking.models import CodeChunk
from app.embeddings.cache import EmbeddingCache
from app.embeddings.models import EmbeddingRecord
from app.embeddings.provider import EmbeddingProvider
from app.embeddings.registry import get_registry

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Main engine for generating embeddings from code chunks.

    The EmbeddingEngine is responsible for:
    1. Receiving a CodeChunk
    2. Checking the cache for existing embeddings
    3. Retrieving the appropriate provider from the registry
    4. Generating embeddings if necessary
    5. Caching new embeddings
    6. Returning EmbeddingRecords

    The engine is designed to be agnostic of:
    - Specific embedding models (delegated to providers)
    - Vector storage (not part of this sprint)
    - Retrieval mechanisms (not part of this sprint)

    Architecture principles:
    - Single Responsibility: Only orchestrates embedding generation
    - Open/Closed: New providers can be added without modifying this class
    - Dependency Inversion: Depends on abstractions, not implementations
    - Interface Segregation: Clean, focused public API
    """

    def __init__(
        self,
        provider_name: Optional[str] = None,
        cache: Optional[EmbeddingCache] = None
    ):
        """Initialize the embedding engine.

        Args:
            provider_name: Name of the provider to use. If None, uses default.
            cache: Cache instance to use. If None, creates a new cache.
        """
        self.registry = get_registry()
        self.provider_name = provider_name
        self._cache = cache or EmbeddingCache()
        self._provider: Optional[EmbeddingProvider] = None

        logger.info(
            f"EmbeddingEngine initialized with provider: "
            f"{provider_name or 'default'}"
        )

    def _get_provider(self) -> EmbeddingProvider:
        """Get or create the embedding provider instance.

        The provider is lazily initialized and reused across embed calls.

        Returns:
            Initialized provider instance

        Raises:
            ValueError: If provider cannot be found or instantiated
        """
        if self._provider is None:
            try:
                provider_class = self.registry.get(self.provider_name)
                self._provider = provider_class()
                logger.info(
                    f"Instantiated provider: {self._provider.get_model_name()}"
                )
            except Exception as e:
                logger.error(f"Failed to instantiate provider: {e}")
                raise ValueError(f"Failed to instantiate provider: {e}") from e

        return self._provider

    def embed(self, chunk: CodeChunk) -> EmbeddingRecord:
        """Generate an embedding for a code chunk.

        This is the main entry point for embedding operations.

        Workflow:
        1. Check cache using content_hash
        2. Return cached embedding if found
        3. Generate new embedding using provider
        4. Cache the new embedding
        5. Return the embedding record

        Args:
            chunk: The code chunk to embed

        Returns:
            EmbeddingRecord with the embedding and metadata

        Raises:
            RuntimeError: If embedding generation fails
        """
        logger.info(f"Embedding chunk: {chunk.id}")

        # Check cache first
        cached_record = self._cache.get(chunk.content_hash)
        if cached_record is not None:
            logger.info(
                f"Using cached embedding for chunk {chunk.id} "
                f"(content_hash: {chunk.content_hash})"
            )
            # Return a fresh record bound to THIS chunk. The cache is keyed
            # only by content_hash, so the cached record may have been created
            # for a different chunk with identical content. Returning it
            # directly would carry that chunk's id and make VectorStore
            # validation fail (record.chunk_id != chunk.id). Reuse the cached
            # vector but rebind identity fields to the current chunk.
            return EmbeddingRecord(
                chunk_id=chunk.id,
                embedding=cached_record.embedding,
                model_name=cached_record.model_name,
                dimension=cached_record.dimension,
                content_hash=chunk.content_hash,
            )

        # Generate new embedding
        logger.info(f"Generating new embedding for chunk {chunk.id}")
        try:
            provider = self._get_provider()
            record = provider.embed(chunk)

            # Cache the new embedding
            self._cache.put(chunk.content_hash, record)

            logger.info(
                f"Embedding generated and cached for chunk {chunk.id} "
                f"(dimension: {record.dimension})"
            )

            return record

        except Exception as e:
            logger.error(f"Failed to embed chunk {chunk.id}: {e}")
            raise RuntimeError(f"Failed to embed chunk: {e}") from e

    def embed_batch(self, chunks: List[CodeChunk]) -> List[EmbeddingRecord]:
        """Generate embeddings for multiple code chunks.

        This method processes chunks sequentially, checking the cache
        for each chunk individually. Future optimizations could batch
        uncached chunks for parallel processing.

        Args:
            chunks: List of code chunks to embed

        Returns:
            List of EmbeddingRecords in the same order as input chunks

        Raises:
            RuntimeError: If any embedding generation fails
        """
        logger.info(f"Embedding batch of {len(chunks)} chunks")

        records = []
        for chunk in chunks:
            try:
                record = self.embed(chunk)
                records.append(record)
            except Exception as e:
                logger.error(f"Failed to embed chunk {chunk.id}: {e}")
                # Continue processing remaining chunks
                continue

        logger.info(
            f"Batch embedding completed: {len(records)}/{len(chunks)} successful"
        )

        return records

    def get_cache_stats(self) -> dict:
        """Get statistics about the embedding cache.

        Returns:
            Dictionary with cache statistics
        """
        return self._cache.get_stats()

    def clear_cache(self) -> None:
        """Clear all cached embeddings.

        This is useful for testing or when memory needs to be freed.
        """
        logger.info("Clearing embedding cache")
        self._cache.clear()

    def get_provider_info(self) -> dict:
        """Get information about the current provider.

        Returns:
            Dictionary with provider information

        Raises:
            RuntimeError: If provider is not initialized
        """
        provider = self._get_provider()
        return {
            "model_name": provider.get_model_name(),
            "dimension": provider.get_dimension(),
            "provider_class": provider.__class__.__name__
        }
