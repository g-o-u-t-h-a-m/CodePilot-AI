"""RAG (Retrieval-Augmented Generation) components.

Sprint 7 provided the retrieval half of the RAG pipeline: converting a
natural-language user question into an embedding and querying the vector
store for the most relevant code chunks.

Sprint 8 adds the generation half: a PromptBuilder converts the retrieved
RetrievalResults into a grounded, budgeted prompt, and a RAGService
orchestrates retrieval -> prompt building -> LLM generation -> grounded
answer, using a pluggable LLMProvider abstraction.

Main components:
    - Retriever: Turns a user question into ranked RetrievalResults
    - RetrievalResult: A single retrieved code chunk with relevance score
    - RAGService: Orchestrates the full pipeline (retrieval + generation)
    - RagResponse: Structured result with grounded answer and provenance

Architecture:
    User question
        -> RAGService.answer()
        -> Retriever (existing)
        -> EmbeddingEngine (existing)
        -> VectorStore abstraction (existing)
        -> RetrievalResult[]
        -> PromptBuilder (Sprint 8)
        -> grounded prompt
        -> LLMProvider (Sprint 8, pluggable)
        -> grounded answer

The Retriever depends only on the EmbeddingEngine and the VectorStore
abstraction; RAGService depends only on the Retriever, PromptBuilder, and
LLMProvider abstractions. Neither knows about ChromaDB or any concrete LLM.

Example usage:
    from app.embeddings import EmbeddingEngine, initialize_providers
    from app.llm import initialize_providers as init_llm, create_llm_provider
    from app.prompts import PromptBuilder
    from app.rag import RAGService, create_retriever
    from app.vectorstore import create_vector_store

    initialize_providers()
    init_llm()
    engine = EmbeddingEngine()
    store = create_vector_store()
    retriever = create_retriever(engine, store)
    builder = PromptBuilder()
    llm = create_llm_provider("mock")

    service = RAGService(
        retriever=retriever,
        prompt_builder=builder,
        llm_provider=llm,
    )
    response = service.answer(
        question="Where is user authentication handled?",
        repository_name="demo-repo",
    )
    print(response.answer)
    print(response.retrieved_results)
"""

import logging

from app.rag.models import RagResponse, RetrievalResult
from app.rag.retriever import Retriever
from app.rag.service import RAGService

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


def create_rag_service(
    retriever: Retriever,
    prompt_builder,
    llm_provider,
    top_k: int = 5,
) -> RAGService:
    """Create a configured RAGService instance.

    Convenience factory that wires the provided Retriever, PromptBuilder,
    and LLMProvider into a RAGService.

    Args:
        retriever: Existing Retriever (from app.rag).
        prompt_builder: A PromptBuilder (from app.prompts).
        llm_provider: An LLMProvider (from app.llm).
        top_k: Default number of retrieval results to fetch.

    Returns:
        A configured RAGService instance
    """
    service = RAGService(
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm_provider=llm_provider,
        top_k=top_k,
    )
    logger.info(f"Created RAG service: {service.__class__.__name__}")
    return service


__all__ = [
    "Retriever",
    "RetrievalResult",
    "RAGService",
    "RagResponse",
    "create_retriever",
    "create_rag_service",
]