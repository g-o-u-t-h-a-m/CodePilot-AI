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
from typing import Dict, List, Optional

from app.chunking.models import CodeChunk
from app.embeddings.models import EmbeddingRecord
from app.vectorstore.models import ChunkEmbeddingPair, SimilarityResult, VectorStoreRecord


class VectorStore(ABC):
    """Abstract base class for vector storage.

    A vector store persists chunk/embedding pairs and their metadata
    so that they can later be retrieved by chunk ID or by semantic
    similarity against a query embedding.

    Implementations must:
    - Persist embeddings, documents, and metadata
    - Support upsert semantics (adding an existing ID must not duplicate)
    - Preserve metadata such as repository_name for later filtering
    - Query by semantic similarity, returning normalized relevance scores
      where higher = more relevant (never exposing raw distance semantics)
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

    @abstractmethod
    def delete_by_repository(self, repository_name: str) -> int:
        """Delete every stored record that belongs to a repository.

        Repository isolation is keyed on the ``repository_name`` metadata
        field. This method removes all records for the repository in a single
        operation so callers (e.g. IndexingService) can replace a repository's
        stale vectors wholesale when re-indexing.

        Args:
            repository_name: The repository whose records should be deleted.

        Returns:
            Number of records deleted. Zero if the repository has no stored
            records (which is not an error).

        Raises:
            RuntimeError: If deletion fails.
        """
        pass

    @abstractmethod
    def count_by_repository(self, repository_name: str) -> int:
        """Count the number of stored records for a repository.

        Repository isolation is keyed on the ``repository_name`` metadata
        field. Examples:

            - An indexed repository can be verified to have as many vectors
              as it produced chunks (``delete_by_repository`` then
              ``add_many`` guarantees this).
            - An existing repository with no vectors can be distinguished
              from a nonexistent one every store can answer.

        Args:
            repository_name: The repository to count records for.

        Returns:
            Number of records currently stored for the repository.

        Raises:
            RuntimeError: If the count fails.
        """
        pass

    @abstractmethod
    def query_similar(
        self,
        embedding: List[float],
        top_k: int,
        filter_metadata: Optional[Dict[str, object]] = None
    ) -> List[SimilarityResult]:
        """Query the store for records most similar to a query embedding.

        This is the semantic retrieval entry point used by the Retriever
        (Sprint 7). Implementations must return records ordered from most
        relevant to least relevant.

        Score semantics are normalized by the store: every returned score
        is in [0, 1] where a higher score means more relevant. Implementations
        must NOT expose raw distance values (e.g. ChromaDB's cosine distance),
        and must not invent conversions that are mathematically invalid for
        the configured metric.

        Args:
            embedding: The query embedding vector
            top_k: Maximum number of results to return
            filter_metadata: Optional metadata filter (e.g. repository_name)
                applied as an exact-match filter before ranking

        Returns:
            List of SimilarityResult ordered from most to least relevant.
            Empty list if no records match.

        Raises:
            ValueError: If embedding is empty or top_k is not positive
            RuntimeError: If the similarity query fails
        """
        pass
