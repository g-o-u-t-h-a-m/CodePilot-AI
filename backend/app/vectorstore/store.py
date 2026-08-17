"""Abstract vector store interface.

This module defines the storage contract that all vector store
implementations must satisfy. It is deliberately independent of
any specific vector database (ChromaDB, Qdrant, FAISS, etc.) so
that the retrieval layer (Sprint 7) can depend on this abstraction
rather than on a concrete implementation.

The interface follows the Dependency Inversion Principle: high-level
modules depend on this abstraction, and low-level implementations
provide the concrete storage behavior.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from app.chunking.models import CodeChunk
from app.embeddings.models import EmbeddingRecord
from app.vectorstore.models import ChunkEmbeddingPair, VectorStoreRecord


class VectorStore(ABC):
    """Abstract base class for vector storage.

    A vector store persists chunk/embedding pairs and their metadata
    so that they can later be retrieved by chunk ID (and, in future
    sprints, by semantic similarity).

    Implementations must:
    - Persist embeddings, documents, and metadata
    - Support upsert semantics (adding an existing ID must not duplicate)
    - Preserve metadata such as repository_name for later filtering
    - Raise meaningful errors instead of swallowing failures

    A vector store MUST NOT generate embeddings itself; it receives
    already-generated EmbeddingRecords from the EmbeddingEngine.
    """

    @abstractmethod
    def add(self, chunk: CodeChunk, record: EmbeddingRecord) -> None:
        """Store a single chunk/embedding pair.

        Args:
            chunk: The source code chunk
            record: The embedding record generated for the chunk

        Raises:
            ValueError: If the embedding record does not belong to the chunk
            RuntimeError: If storage fails
        """
        pass

    @abstractmethod
    def add_many(self, pairs: List[ChunkEmbeddingPair]) -> int:
        """Store multiple chunk/embedding pairs in a single batch.

        Args:
            pairs: List of chunk/embedding pairs to store

        Returns:
            Number of pairs successfully stored

        Raises:
            RuntimeError: If batch storage fails
        """
        pass

    @abstractmethod
    def get(self, chunk_id: str, include_embedding: bool = True) -> Optional[VectorStoreRecord]:
        """Retrieve a stored record by chunk ID.

        Args:
            chunk_id: The chunk ID to look up
            include_embedding: Whether to include the embedding vector

        Returns:
            The stored record, or None if not found
        """
        pass

    @abstractmethod
    def delete(self, chunk_id: str) -> bool:
        """Delete a stored record by chunk ID.

        Args:
            chunk_id: The chunk ID to delete

        Returns:
            True if a record was deleted, False if it did not exist
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """Count the number of stored records.

        Returns:
            Total number of records currently stored
        """
        pass
