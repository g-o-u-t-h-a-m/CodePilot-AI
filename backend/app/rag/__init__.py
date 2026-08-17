"""RAG (Retrieval-Augmented Generation) components.

Sprint 7 provides the retrieval half of the RAG pipeline: converting a
natural-language user question into an embedding and querying the vector
store for the most relevant code chunks. Later sprints add the generation
half (prompt building, LLM calls).

Main components:
    - Retriever: Turns a user question into ranked RetrievalResults
    - RetrievalResult: A single retrieved code chunk with relevance score

Architecture:
    User question
        -> Retriever
        -> EmbeddingEngine (existing)
        -> question embedding
        -> VectorStore abstraction (existing)
        -> ChromaDB
        -> Top-K relevant chunks
        -> RetrievalResult[]

The Retriever depends only on the EmbeddingEngine and the VectorStore
abstraction; it never depends on ChromaVectorStore or ChromaDB directly.

Example usage:
    from app.embeddings import EmbeddingEngine, initialize_providers
    from app.rag import Retriever
    from app.vectorstore import create_vector_store

    initialize_providers()
    engine = EmbeddingEngine()
    store = create_vector_store()
    retriever = Retriever(embedding_engine=engine, vector_store=store)

    results = retriever.retrieve(
        question="How is user authentication handled?",
        repository_name="demo-repo",
        top_k=5
    )
    for result in results:
        print(result.relative_path, result.relevance_score)
"""

import logging

from app.rag.models import RetrievalResult
from app.rag.retriever import Retriever

logger = logging.getLogger(__name__)


def create_retriever(embedding_engine, vector_store) -> Retriever:
    """Create a configured Retriever instance.

    Convenience factory that wires the provided EmbeddingEngine and
    VectorStore abstraction into a Retriever.

    Args:
        embedding_engine: Existing EmbeddingEngine (from app.embeddings)
        vector_store: A VectorStore implementation (from app.vectorstore)

    Returns:
        A configured Retriever instance
    """
    retriever = Retriever(
        embedding_engine=embedding_engine,
        vector_store=vector_store
    )
    logger.info(f"Created retriever: {retriever.__class__.__name__}")
    return retriever


__all__ = [
    "Retriever",
    "RetrievalResult",
    "create_retriever",
]
