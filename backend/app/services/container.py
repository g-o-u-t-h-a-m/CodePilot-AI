"""Central dependency container for the service layer.

This module is the single place where services and their dependencies are
constructed, so API route functions never instantiate engines, stores, or
providers directly. It reuses the existing factories and registries from
Sprints 4-8:

    Indexing:  RepositoryScanner, ChunkEngine, EmbeddingEngine,
               create_vector_store(), RepositoryManager
    Query:     EmbeddingEngine, create_vector_store(), create_retriever(),
               create_prompt_builder(), create_llm_provider(),
               create_rag_service()

Construction is lazy and cached: the first access builds the component
(initializing registries and loading the embedding model), and subsequent
accesses reuse the same instance. This keeps the 0-budget mock provider
default and honors the vector store configuration from the environment
(CHROMA_PERSIST_DIR / COLLECTION_NAME).

Dependency Inversion:
    Consumers receive the VectorStore abstraction, the Retriever, the
    LLMProvider, and the RAGService through these singletons. The container
    is the only place that knows the concrete ChromaVectorStore and the
    concrete LLM provider selection.
"""

import logging
import os
from typing import Optional

from app.chunking import ChunkEngine, initialize_strategies
from app.embeddings import EmbeddingEngine, initialize_providers as init_embeddings
from app.llm import create_llm_provider, initialize_providers as init_llm
from app.prompts import create_prompt_builder
from app.rag import create_rag_service, create_retriever
from app.repository.git_manager import RepositoryManager
from app.repository.scanner import RepositoryScanner
from app.services.indexing_service import IndexingService
from app.vectorstore import create_vector_store

logger = logging.getLogger(__name__)

# Directory (relative to the backend working directory) where cloned
# repositories are stored, matching RepositoryService/RepositoryManager.
_REPOSITORIES_DIR = "indexed_repos"

# Default ChromaDB collection name used by the vector store factory.
_DEFAULT_COLLECTION_NAME = "code_chunks"


def _configured_chroma_path() -> Optional[str]:
    """Resolve the ChromaDB persistence directory from the environment.

    Honors ``CHROMA_PERSIST_DIR`` if set (see .env.example). When unset,
    ``create_vector_store`` falls back to the module default
    (``backend/chroma_db``).

    Returns:
        Absolute path when CHROMA_PERSIST_DIR is set, otherwise None.
    """
    raw = os.getenv("CHROMA_PERSIST_DIR")
    if not raw:
        return None
    return os.path.abspath(raw)


def _configured_collection_name() -> str:
    """Resolve the ChromaDB collection name from the environment.

    Honors ``COLLECTION_NAME`` if set (see .env.example), otherwise uses the
    vector store default.

    Returns:
        The configured collection name.
    """
    name = os.getenv("COLLECTION_NAME")
    return name if name else _DEFAULT_COLLECTION_NAME


class Container:
    """Lazily constructs and caches the application's services.

    Each component is built on first access and reused afterwards. No
    component is constructed at import time, so importing the container
    never loads the embedding model or touches ChromaDB.
    """

    def __init__(self):
        """Initialize the container with no constructed components."""
        self._scanner: Optional[RepositoryScanner] = None
        self._chunk_engine: Optional[ChunkEngine] = None
        self._embedding_engine: Optional[EmbeddingEngine] = None
        self._vector_store = None
        self._repository_manager: Optional[RepositoryManager] = None
        self._indexing_service: Optional[IndexingService] = None
        self._rag_service = None
        logger.info("Container initialized (lazy)")

    # ------------------------------------------------------------------
    # Component accessors (lazy singletons)
    # ------------------------------------------------------------------

    @property
    def scanner(self) -> RepositoryScanner:
        """Return the shared RepositoryScanner."""
        if self._scanner is None:
            self._scanner = RepositoryScanner()
        return self._scanner

    @property
    def chunk_engine(self) -> ChunkEngine:
        """Return the shared ChunkEngine (strategies registered once)."""
        if self._chunk_engine is None:
            initialize_strategies()
            self._chunk_engine = ChunkEngine()
        return self._chunk_engine

    @property
    def embedding_engine(self) -> EmbeddingEngine:
        """Return the shared EmbeddingEngine (providers registered once)."""
        if self._embedding_engine is None:
            init_embeddings()
            self._embedding_engine = EmbeddingEngine()
        return self._embedding_engine

    @property
    def vector_store(self):
        """Return the shared VectorStore abstraction.

        Uses the existing ``create_vector_store`` factory so the concrete
        implementation and its persistence location are chosen in exactly
        one place. Configuration is read from the environment
        (CHROMA_PERSIST_DIR / COLLECTION_NAME).
        """
        if self._vector_store is None:
            self._vector_store = create_vector_store(
                persistence_path=_configured_chroma_path(),
                collection_name=_configured_collection_name(),
            )
        return self._vector_store

    @property
    def repository_manager(self) -> RepositoryManager:
        """Return the shared RepositoryManager.

        Uses the same relative base path as RepositoryService so the index
        flow and the clone flow share one repository layout.
        """
        if self._repository_manager is None:
            self._repository_manager = RepositoryManager(
                base_path=_REPOSITORIES_DIR
            )
        return self._repository_manager

    @property
    def indexing_service(self) -> IndexingService:
        """Return the shared IndexingService."""
        if self._indexing_service is None:
            self._indexing_service = IndexingService(
                scanner=self.scanner,
                chunk_engine=self.chunk_engine,
                embedding_engine=self.embedding_engine,
                vector_store=self.vector_store,
                repository_manager=self.repository_manager,
            )
        return self._indexing_service

    @property
    def rag_service(self):
        """Return the shared RAGService (the Sprint 8 service, reused).

        Wires the existing EmbeddingEngine, VectorStore, Retriever,
        PromptBuilder, and LLMProvider using the existing factories. The
        active LLM provider is selected by configuration (LLM_PROVIDER),
        defaulting to the 0-cost mock provider.
        """
        if self._rag_service is None:
            init_llm()
            retriever = create_retriever(
                embedding_engine=self.embedding_engine,
                vector_store=self.vector_store,
            )
            prompt_builder = create_prompt_builder()
            llm_provider = create_llm_provider()
            self._rag_service = create_rag_service(
                retriever=retriever,
                prompt_builder=prompt_builder,
                llm_provider=llm_provider,
            )
        return self._rag_service


# Module-level shared container so routers can request services without
# constructing them and without repeating the wiring.
container = Container()


__all__ = ["Container", "container"]
