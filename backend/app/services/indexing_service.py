"""Indexing orchestration service.

IndexingService is the service-layer entry point for indexing an
already-cloned repository. It orchestrates the existing domain components
without re-implementing any of their logic:

    RepositoryScanner -> SourceFile[] -> ChunkEngine -> CodeChunk[]
        -> EmbeddingEngine -> EmbeddingRecord[] -> VectorStore -> ChromaDB

Re-indexing safety (stale vectors):
    CodeChunk IDs default to a random ``uuid4``. Re-running the pipeline for
    the same chunk content would mint NEW IDs and, because ChromaDB keys on
    the ID, create duplicate vectors. This service therefore reassigns each
    chunk a deterministic ID derived from ``repository_name`` +
    ``relative_path`` + ``chunk_index``, so the same chunk always maps to the
    same vector ID. Deterministic IDs alone are not enough, though: a file
    that shrinks below its old chunk count (or a deleted file) would leave
    the now-unproducible chunks behind. Indexing is therefore FULL
    REPLACEMENT per repository: once scanning, chunking, and embedding have
    all succeeded, the service asks the VectorStore to delete the
    repository's previous vectors, then stores the freshly generated ones.
    Stale vectors for deleted or shrunk files cannot survive a re-index, and
    a failed pipeline stage never erases a previously valid index because
    the deletion happens only after every upstream stage succeeded.

Paths:
    The repository layout follows the existing convention established by
    RepositoryService/RepositoryManager: ``indexed_repos/<name>/source``.
    No hard-coded absolute paths are used; the base directory is resolved
    relative to the backend working directory (``backend/``), matching the
    clone flow.
"""

import hashlib
import logging
import time
from pathlib import Path
from typing import List

from app.chunking.engine import ChunkEngine
from app.embeddings.engine import EmbeddingEngine
from app.models.indexing import IndexRepositoryRequest, IndexRepositoryResponse
from app.models.source_file import SourceFile
from app.repository.git_manager import RepositoryManager
from app.repository.scanner import RepositoryScanner
from app.vectorstore.models import ChunkEmbeddingPair
from app.vectorstore.store import VectorStore

logger = logging.getLogger(__name__)


class RepositoryUnavailableError(Exception):
    """Raised when a repository directory exists but is not a valid clone.

    Maps to HTTP 409, indicating the client must re-clone the repository.
    """


class RepositoryMissingError(Exception):
    """Raised when the requested repository is not present locally.

    The index endpoint never clones; callers must clone first via
    ``POST /repository/clone``, so this error maps to HTTP 404.
    """


class RepositoryEmptyError(Exception):
    """Raised when the repository contains no supported source files.

    Maps to HTTP 422: nothing can be indexed for an empty repository.
    """


class IndexingError(Exception):
    """Raised when a stage of the indexing pipeline fails.

    The message is intentionally generic so no internal details (paths,
    stack traces, model internals) leak to API clients.
    """


class IndexingService:
    """Orchestrates scanning, chunking, embedding, and storage.

    Responsibilities:
        - Locate an already-cloned repository using the existing
          RepositoryManager layout.
        - Delegate scanning to the existing RepositoryScanner.
        - Delegate chunking to the existing ChunkEngine.
        - Delegate embedding to the existing EmbeddingEngine.
        - Store via the existing VectorStore abstraction, replacing the
          repository's previous vectors so stale chunks (deleted files,
          shrunk files) cannot survive a re-index.
        - Return structured indexing statistics.

    SOLID notes:
        - Single Responsibility: only orchestrates indexing.
        - Dependency Inversion: depends on abstractions (VectorStore) and
          the existing engines, never on ChromaDB or a concrete store.
    """

    def __init__(
        self,
        scanner: RepositoryScanner,
        chunk_engine: ChunkEngine,
        embedding_engine: EmbeddingEngine,
        vector_store: VectorStore,
        repository_manager: RepositoryManager,
    ):
        """Initialize the indexing service with its dependencies.

        Every component is injected; the service constructs none of them.
        Dependency construction lives in the service container
        (``app.services.container``), so API routes never instantiate
        engines or the vector store directly.

        Args:
            scanner: Existing RepositoryScanner.
            chunk_engine: Existing ChunkEngine (strategies initialized).
            embedding_engine: Existing EmbeddingEngine (providers initialized).
            vector_store: VectorStore abstraction (e.g. ChromaVectorStore).
            repository_manager: Existing RepositoryManager for locating the
                already-cloned repository.
        """
        self._scanner = scanner
        self._chunk_engine = chunk_engine
        self._embedding_engine = embedding_engine
        self._vector_store = vector_store
        self._repository_manager = repository_manager
        logger.info(
            "IndexingService initialized "
            f"(store: {vector_store.__class__.__name__})"
        )

    def index_repository(
        self, request: IndexRepositoryRequest
    ) -> IndexRepositoryResponse:
        """Index an already-cloned repository end to end.

        Args:
            request: Validated index request containing the repository name.

        Returns:
            Structured statistics for the indexing operation.

        Raises:
            RepositoryMissingError: If the repository is not cloned locally.
                Maps to HTTP 404.
            RepositoryEmptyError: If the repository has no supported files.
                Maps to HTTP 422.
            IndexingError: If a pipeline stage fails. Maps to HTTP 500.
        """
        repo_name = request.repository_name.strip()
        repo_source_dir = self._locate_repository_source(repo_name)
        started = time.time()

        logger.info(
            f"Starting indexing for repository '{repo_name}' "
            f"at {repo_source_dir}"
        )

        # 1. Scan the repository (existing RepositoryScanner).
        try:
            source_files: List[SourceFile] = self._scanner.scan(str(repo_source_dir))
        except Exception as e:
            logger.error(f"Scanner failed for '{repo_name}': {e}", exc_info=True)
            raise IndexingError(
                f"Failed to scan repository"
            ) from e

        if not source_files:
            logger.warning(f"Repository '{repo_name}' contains no supported files")
            raise RepositoryEmptyError(
                f"Repository '{repo_name}' contains no supported source files"
            )

        # 2. Chunk every source file (existing ChunkEngine).
        try:
            chunks = self._chunk_engine.chunk_files(source_files)
        except Exception as e:
            logger.error(f"Chunking failed for '{repo_name}': {e}", exc_info=True)
            raise IndexingError(
                f"Chunking failed for repository"
            ) from e

        if not chunks:
            logger.warning(f"No chunks generated for repository '{repo_name}'")
            raise RepositoryEmptyError(
                f"No chunks could be generated for repository '{repo_name}'"
            )

        # Deterministic IDs so re-indexing upserts instead of duplicating.
        self._assign_deterministic_ids(repo_name, chunks)

        # 3. Generate embeddings (existing EmbeddingEngine).
        try:
            embedding_records = self._embedding_engine.embed_batch(chunks)
        except Exception as e:
            logger.error(f"Embedding failed for '{repo_name}': {e}", exc_info=True)
            raise IndexingError(
                f"Embedding generation failed for repository"
            ) from e

        if not embedding_records:
            logger.error(f"No embeddings generated for repository '{repo_name}'")
            raise IndexingError(
                f"Embedding generation failed for repository"
            )

        # 4. Replace the repository's existing vectors now that scanning,
        #    chunking, and embedding have all succeeded. The stale chunks
        #    (deleted files, files that shrank below their old chunk count,
        #    files whose chunk boundaries changed) are removed so the stored
        #    state exactly matches the current repository. Because this
        #    happens only after every upstream stage succeeded, a failed
        #    indexing operation never erases a previously valid index.
        try:
            deleted = self._vector_store.delete_by_repository(repo_name)
        except Exception as e:
            logger.error(
                f"Stale-vector deletion failed for '{repo_name}': {e}",
                exc_info=True,
            )
            raise IndexingError(
                f"Vector storage failed for repository"
            ) from e
        logger.info(
            f"Removed {deleted} stale vectors for repository '{repo_name}'"
        )

        # 5. Store chunk/embedding pairs (existing VectorStore, upsert).
        pairs = [
            ChunkEmbeddingPair(chunk=chunk, record=record)
            for chunk, record in zip(chunks, embedding_records)
        ]

        try:
            stored = self._vector_store.add_many(pairs)
        except Exception as e:
            logger.error(f"Vector storage failed for '{repo_name}': {e}", exc_info=True)
            raise IndexingError(
                f"Vector storage failed for repository"
            ) from e

        duration = time.time() - started

        try:
            collection_count = self._vector_store.count()
        except Exception as e:
            logger.error(f"Vector count failed for '{repo_name}': {e}", exc_info=True)
            collection_count = stored

        try:
            # count_by_repository is exact after the replacement above:
            # it equals ``stored`` because the repository's pre-existing
            # vectors were deleted and none can linger as stale chunks.
            repository_vectors = self._vector_store.count_by_repository(repo_name)
        except Exception as e:
            logger.error(
                f"Repository vector count failed for '{repo_name}': {e}",
                exc_info=True,
            )
            repository_vectors = stored

        logger.info(
            f"Indexing complete for '{repo_name}': "
            f"{len(source_files)} files, {len(chunks)} chunks, "
            f"{len(embedding_records)} embeddings, {stored} vectors "
            f"(repository: {repository_vectors}, collection: {collection_count}) "
            f"in {duration:.3f}s"
        )

        # Include embedding model metadata for the response when available.
        embedding_model = None
        try:
            info = self._embedding_engine.get_provider_info()
            embedding_model = info.get("model_name")
        except Exception as e:
            logger.debug(f"Could not read embedding provider info: {e}")

        return IndexRepositoryResponse(
            success=True,
            repository_name=repo_name,
            files_scanned=len(source_files),
            chunks_generated=len(chunks),
            embeddings_generated=len(embedding_records),
            vectors_stored=stored,
            collection_count=collection_count,
            repository_vectors=repository_vectors,
            duration=round(duration, 4),
            embedding_model=embedding_model,
            message=f"Successfully indexed repository '{repo_name}'",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _locate_repository_source(self, repo_name: str) -> Path:
        """Resolve the source directory of an already-cloned repository.

        Uses the existing RepositoryManager layout
        (``indexed_repos/<name>/source``). A repository is considered
        present only when it was cloned through the manager, i.e. it has a
        ``.git`` directory. This deliberately mirrors
        ``RepositoryManager.repository_exists`` so the index endpoint only
        ever indexes what the clone flow created.

        Args:
            repo_name: Trimmed repository name.

        Returns:
            Path to the repository source directory.

        Raises:
            RepositoryMissingError: If the repository is not present locally.
            RepositoryUnavailableError: If a directory exists but is not a
                valid cloned repository.
        """
        repo_dir = self._repository_manager.base_path / repo_name
        source_dir = repo_dir / "source"
        git_dir = source_dir / ".git"

        if not source_dir.is_dir():
            raise RepositoryMissingError(
                f"Repository '{repo_name}' is not present locally. "
                f"Clone it first via POST /repository/clone."
            )

        if not git_dir.exists():
            raise RepositoryUnavailableError(
                f"Repository directory '{repo_name}' exists but is not a "
                f"cloned repository. Re-run POST /repository/clone."
            )

        return source_dir

    @staticmethod
    def _deterministic_chunk_id(repo_name: str, relative_path: str, chunk_index: int) -> str:
        """Produce a stable chunk ID for a given chunk location.

        The ID is derived solely from stable identifiers (repository name,
        relative file path, and chunk index) so that re-running the pipeline
        for unchanged content yields the same ID, enabling upsert-safe
        re-indexing.

        Args:
            repo_name: Repository name.
            relative_path: File path relative to the repository root.
            chunk_index: 0-based chunk index within the file.

        Returns:
            A deterministic, collision-resistant ID string.
        """
        raw = f"{repo_name}::{relative_path}::{chunk_index}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"idx-{digest[:40]}"

    @classmethod
    def _assign_deterministic_ids(
        cls, repo_name: str, chunks: List
    ) -> None:
        """Overwrite random chunk IDs with deterministic IDs in place.

        Args:
            repo_name: Repository being indexed.
            chunks: The CodeChunk list for this repository. Each chunk's
                ``id`` attribute is reassigned; ``content_hash`` and other
                fields are left untouched.
        """
        for chunk in chunks:
            chunk.id = cls._deterministic_chunk_id(
                repo_name, chunk.relative_path, chunk.chunk_index
            )
        logger.info(f"Assigned deterministic chunk IDs for '{repo_name}'")


__all__ = [
    "IndexingService",
    "RepositoryMissingError",
    "RepositoryUnavailableError",
    "RepositoryEmptyError",
    "IndexingError",
]