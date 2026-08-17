"""Vector store module for CodePilot AI.

This module provides a clean abstraction for persisting chunk/embedding
pairs so they can be retrieved later (by the Retriever in Sprint 7).

Main components:
    - VectorStore: Abstract interface for storage operations
    - ChromaVectorStore: ChromaDB-backed persistent implementation
    - VectorStoreRecord: Application-facing stored record
    - ChunkEmbeddingPair: Chunk/embedding pair for batch insertion

The interface exposes repository-scoped operations (``delete_by_repository``,
``count_by_repository``) so consumers can isolate, inspect, and replace one
repository's vectors (e.g. full replacement on re-index) without depending on
ChromaDB.

Design patterns:
    - Strategy Pattern: Different stores implement the same interface
    - Dependency Inversion: Consumers depend on VectorStore, not ChromaDB

Example usage:
    from app.chunking.models import CodeChunk
    from app.embeddings import EmbeddingEngine, initialize_providers
    from app.vectorstore import ChromaVectorStore

    # Generate an embedding for a chunk
    initialize_providers()
    engine = EmbeddingEngine()
    record = engine.embed(chunk)

    # Store the chunk/embedding pair persistently
    store = ChromaVectorStore()
    store.add(chunk, record)

    # Retrieve it later by chunk ID
    stored = store.get(chunk.id)
    print(stored.document)
"""

import logging

from app.vectorstore.chroma_store import (
    DEFAULT_CHROMA_PATH,
    DEFAULT_COLLECTION_NAME,
    ChromaVectorStore,
)
from app.vectorstore.models import (
    ChunkEmbeddingPair,
    SimilarityResult,
    VectorStoreRecord,
)
from app.vectorstore.store import VectorStore

logger = logging.getLogger(__name__)


def create_vector_store(
    persistence_path=None,
    collection_name=DEFAULT_COLLECTION_NAME
) -> VectorStore:
    """Create a default vector store instance.

    Convenience factory that returns a ChromaVectorStore. Consumers
    that depend only on the VectorStore interface can use this to
    avoid coupling to the concrete implementation.

    Args:
        persistence_path: Directory for persistent storage.
            Defaults to backend/chroma_db.
        collection_name: Name of the collection to use.

    Returns:
        A configured VectorStore instance
    """
    store = ChromaVectorStore(
        persistence_path=persistence_path,
        collection_name=collection_name
    )
    logger.info(f"Created vector store: {store.__class__.__name__}")
    return store


__all__ = [
    "VectorStore",
    "ChromaVectorStore",
    "VectorStoreRecord",
    "ChunkEmbeddingPair",
    "SimilarityResult",
    "create_vector_store",
    "DEFAULT_CHROMA_PATH",
    "DEFAULT_COLLECTION_NAME",
]
