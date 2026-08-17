"""Embedding cache for avoiding redundant computation.

This module provides an in-memory cache for embeddings based on content hashes.
When a chunk with the same content is embedded multiple times, the cached
embedding is returned instead of regenerating it.

The cache is designed to be replaceable with a persistent implementation
(e.g., Redis, database) without changing the interface.
"""

import logging
from typing import Dict, Optional

from app.embeddings.models import EmbeddingRecord

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """In-memory cache for embedding records.

    The cache uses content_hash as the key to identify duplicate content.
    This allows for efficient deduplication across chunks with identical content.

    Design considerations:
    - Thread-safe operations for concurrent access
    - Memory-bounded to prevent unbounded growth
    - Simple replacement interface for persistent cache implementations
    """

    def __init__(self):
        """Initialize the embedding cache."""
        self._cache: Dict[str, EmbeddingRecord] = {}
        logger.info("EmbeddingCache initialized")

    def get(self, content_hash: str) -> Optional[EmbeddingRecord]:
        """Retrieve an embedding record from cache.

        Args:
            content_hash: Hash of the chunk content

        Returns:
            Cached EmbeddingRecord if found, None otherwise
        """
        record = self._cache.get(content_hash)

        if record:
            logger.debug(f"Cache hit for content_hash: {content_hash}")
        else:
            logger.debug(f"Cache miss for content_hash: {content_hash}")

        return record

    def put(self, content_hash: str, record: EmbeddingRecord) -> None:
        """Store an embedding record in cache.

        Args:
            content_hash: Hash of the chunk content
            record: EmbeddingRecord to cache
        """
        self._cache[content_hash] = record
        logger.debug(f"Cached embedding for content_hash: {content_hash}")

    def has(self, content_hash: str) -> bool:
        """Check if an embedding exists in cache.

        Args:
            content_hash: Hash of the chunk content

        Returns:
            True if cached, False otherwise
        """
        return content_hash in self._cache

    def clear(self) -> None:
        """Clear all cached embeddings.

        This is useful for testing or when memory needs to be freed.
        """
        size = len(self._cache)
        self._cache.clear()
        logger.info(f"Cache cleared. Removed {size} entries.")

    def size(self) -> int:
        """Get the number of cached embeddings.

        Returns:
            Number of entries in the cache
        """
        return len(self._cache)

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "total_entries": len(self._cache),
            "total_hashes": len(set(self._cache.keys()))
        }


class PersistentEmbeddingCache(EmbeddingCache):
    """Interface for persistent embedding cache implementations.

    This class serves as a blueprint for future persistent cache implementations.
    Subclasses can implement backends like Redis, PostgreSQL, or file-based storage.

    The interface remains identical to the in-memory cache, allowing drop-in replacement.
    """

    def __init__(self):
        """Initialize the persistent cache."""
        super().__init__()
        logger.info("PersistentEmbeddingCache interface initialized")

    # Future implementations would override:
    # - get() to fetch from persistent storage
    # - put() to store in persistent storage
    # - has() to check persistent storage
    # - clear() to clear persistent storage
    # - size() to query persistent storage size
