"""Semantic retriever for RAG over code chunks.

The Retriever is the retrieval entry point for the RAG pipeline. It
converts a natural-language user question into an embedding using the
existing EmbeddingEngine, queries the existing VectorStore abstraction
for the most relevant code chunks, and returns structured
RetrievalResult objects.

Dependency Inversion:
    The Retriever depends on the EmbeddingEngine and the VectorStore
    abstraction (both injected through the constructor). It never
    depends on ChromaVectorStore or ChromaDB directly, so the vector
    database can be swapped without touching this module.

Data flow:
    User question
        -> EmbeddingEngine (existing)
        -> question embedding
        -> VectorStore.query_similar (existing abstraction)
        -> top-K SimilarityResults
        -> RetrievalResult[]

Reuse of the existing embedding pipeline:
    The EmbeddingEngine.embed() API operates on CodeChunks (content
    hashing, caching, provider abstraction). To embed a free-form
    question without introducing a second embedding model, the
    Retriever wraps the question text in a synthetic CodeChunk and
    delegates to the same engine. This reuses the provider registry,
    the embedding cache, and the BGE model exactly as Sprint 5 uses
    them for code chunks.
"""

import hashlib
import logging
from typing import List, Optional

from app.chunking.models import ChunkType, CodeChunk
from app.embeddings.engine import EmbeddingEngine
from app.rag.models import RetrievalResult
from app.vectorstore.store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieves relevant code chunks for a natural-language question.

    The Retriever orchestrates question embedding and vector similarity
    search. It is intentionally unaware of any concrete vector database:
    similarity querying is delegated to the injected VectorStore.

    Responsibilities:
        - Validate user input with meaningful error messages
        - Embed the question with the existing EmbeddingEngine
        - Query the VectorStore abstraction, filtered by repository_name
        - Normalize store results into RetrievalResult objects
        - Preserve all metadata returned by the store
        - Return results ordered from most to least relevant

    SOLID notes:
        - Single Responsibility: only semantic retrieval
        - Open/Closed: extendable without modifying its dependencies
        - Dependency Inversion: depends on abstractions (engine, store)
    """

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        vector_store: VectorStore
    ):
        """Initialize the retriever with its dependencies.

        Args:
            embedding_engine: Existing EmbeddingEngine used to embed
                the question. Must be configured with a registered
                provider (call initialize_providers() first).
            vector_store: VectorStore abstraction used for similarity
                querying. ChromaVectorStore is injected here, but the
                retriever only relies on the VectorStore interface.

        Raises:
            TypeError: If a dependency is missing or of the wrong type
        """
        if embedding_engine is None:
            raise TypeError("embedding_engine is required")
        if vector_store is None:
            raise TypeError("vector_store is required")
        if not isinstance(embedding_engine, EmbeddingEngine):
            raise TypeError(
                f"embedding_engine must be an EmbeddingEngine, "
                f"got {type(embedding_engine).__name__}"
            )
        if not isinstance(vector_store, VectorStore):
            raise TypeError(
                f"vector_store must implement VectorStore, "
                f"got {type(vector_store).__name__}"
            )

        self._engine = embedding_engine
        self._store = vector_store
        logger.info(
            f"Retriever initialized (engine: {embedding_engine.__class__.__name__}, "
            f"store: {vector_store.__class__.__name__})"
        )

    def retrieve(
        self,
        question: str,
        repository_name: str,
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """Retrieve the most relevant code chunks for a question.

        Args:
            question: Natural-language user question (non-empty)
            repository_name: Repository to search within (non-empty).
                Used as a metadata filter so retrieval never mixes
                repositories.
            top_k: Maximum number of results to return (positive)

        Returns:
            List of RetrievalResult ordered from most relevant to least
            relevant. Empty list if no chunks match the question.

        Raises:
            ValueError: If question or repository_name is empty/whitespace,
                or if top_k is not positive
            RuntimeError: If embedding generation or the similarity
                query fails
        """
        question = self._validate_question(question)
        repository_name = self._validate_repository_name(repository_name)
        self._validate_top_k(top_k)

        logger.info(
            f"Retrieving top {top_k} chunks for repository "
            f"'{repository_name}': {question!r}"
        )

        # 1. Convert the question into an embedding using the existing engine.
        question_record = self._embed_question(question)

        # 2. Query the VectorStore abstraction, isolated to the repository.
        similar = self._store.query_similar(
            embedding=question_record.embedding,
            top_k=top_k,
            filter_metadata={"repository_name": repository_name}
        )

        # 3. Normalize store results into application-facing results.
        results = [self._to_retrieval_result(item) for item in similar]
        logger.info(f"Retrieval returned {len(results)} results")
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_question(self, question: str):
        """Embed a free-form question using the existing EmbeddingEngine.

        The engine's embed() API is chunk-based, so the question is
        wrapped in a synthetic CodeChunk. This intentionally reuses the
        existing provider + cache pipeline rather than introducing a
        second embedding model.

        Args:
            question: The trimmed, validated question text

        Returns:
            The EmbeddingRecord produced by the engine

        Raises:
            RuntimeError: If embedding generation fails
        """
        question_chunk = CodeChunk(
            repository_name="_query",
            relative_path="_query.txt",
            language="Text",
            chunk_type=ChunkType.GENERIC,
            chunk_index=0,
            start_line=1,
            end_line=max(1, question.count("\n") + 1),
            content=question,
            content_hash=hashlib.sha256(question.encode("utf-8")).hexdigest(),
            metadata={"is_query": True}
        )
        logger.debug(f"Embedding question via existing EmbeddingEngine")
        try:
            record = self._engine.embed(question_chunk)
            logger.info(
                f"Question embedded (dimension: {record.dimension}, "
                f"model: {record.model_name})"
            )
            return record
        except Exception as e:
            logger.error(f"Failed to embed question: {e}")
            raise RuntimeError(f"Failed to embed question: {e}") from e

    def _to_retrieval_result(self, similar):
        """Convert a store SimilarityResult into a RetrievalResult.

        All metadata is preserved verbatim. Fields required by the
        RetrievalResult model are read from the stored metadata where
        available, falling back to safe defaults so a result is never
        dropped because of a missing key.

        Args:
            similar: A SimilarityResult from the vector store

        Returns:
            A populated RetrievalResult
        """
        metadata = dict(similar.metadata or {})

        return RetrievalResult(
            chunk_id=similar.id,
            content=similar.document or "",
            repository_name=str(metadata.get("repository_name", "")),
            relative_path=str(metadata.get("relative_path", "")),
            language=str(metadata.get("language", "")),
            chunk_type=metadata.get("chunk_type", ChunkType.GENERIC),
            chunk_index=metadata.get("chunk_index", 0),
            start_line=metadata.get("start_line", 1),
            end_line=metadata.get("end_line", 1),
            relevance_score=similar.score,
            metadata=metadata
        )

    @staticmethod
    def _validate_question(question: str) -> str:
        """Validate the question is non-empty text.

        Args:
            question: Raw question input

        Returns:
            The trimmed question

        Raises:
            ValueError: If the question is empty or whitespace-only
        """
        if not isinstance(question, str):
            raise ValueError(
                f"question must be a string, got {type(question).__name__}"
            )
        trimmed = question.strip()
        if not trimmed:
            raise ValueError("question must not be empty or whitespace")
        return trimmed

    @staticmethod
    def _validate_repository_name(repository_name: str) -> str:
        """Validate the repository name is non-empty text.

        Args:
            repository_name: Raw repository name input

        Returns:
            The trimmed repository name

        Raises:
            ValueError: If the name is empty or whitespace-only
        """
        if not isinstance(repository_name, str):
            raise ValueError(
                f"repository_name must be a string, "
                f"got {type(repository_name).__name__}"
            )
        trimmed = repository_name.strip()
        if not trimmed:
            raise ValueError("repository_name must not be empty or whitespace")
        return trimmed

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        """Validate that top_k is a positive integer.

        Args:
            top_k: Requested result count

        Raises:
            ValueError: If top_k is not a positive integer
        """
        if isinstance(top_k, bool):
            raise ValueError(f"top_k must be a positive integer, got {top_k}")
        if not isinstance(top_k, int):
            raise ValueError(
                f"top_k must be an integer, got {type(top_k).__name__}"
            )
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")


__all__ = ["Retriever"]
