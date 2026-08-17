"""RAG query API router.

Exposes the Sprint 8 RAG pipeline over HTTP. The route function contains no
RAG logic: it validates the request via Pydantic, establishes the repository
semantics below, delegates the question to the existing RAGService, and maps
the response to a client-facing model. The service pipeline is:

    API -> RAGService -> Retriever -> VectorStore/ChromaDB
        -> PromptBuilder -> LLMProvider -> RagResponse

Repository semantics (client-facing):
    The endpoint distinguishes three situations so callers can tell "nothing
    can be answered" from "no relevant context":

    A) The repository does not exist locally -> HTTP 404. Existence reuses
       the existing RepositoryManager clone layout.
    B) The repository exists locally but has not been indexed -> HTTP 404.
       Indexed-ness is decided by the VectorStore abstraction
       (``count_by_repository``); the store is the source of truth for what
       was actually persisted, so no duplicate RAG logic is introduced.
       Both 404 responses use a consistent "not found or has not been
       indexed" message: the distinction between (A) and (B) is documented
       behavior for the client, not a leak of internal machinery.
    C) The repository is indexed but the question has no relevant context
       -> HTTP 200 with ``insufficient_context=true`` and no fabricated
       repository-specific answer (existing RAGService no-fabrication
       behavior). This is not conflated with a repository that does not
       exist.
"""

import logging
from fastapi import APIRouter, HTTPException, status

from app.models.rag_query import (
    RagQueryRequest,
    RagQueryResponse,
    RagQuerySource,
)
from app.services.container import container

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post(
    "/query",
    response_model=RagQueryResponse,
    responses={
        status.HTTP_404_NOT_FOUND: {
            "description": (
                "Repository does not exist locally, or exists locally but "
                "has not been indexed"
            )
        },
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "description": "Invalid request (empty question/repository, invalid top_k)"
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Retrieval or generation failure"
        },
    },
)
async def query_rag(request: RagQueryRequest) -> RagQueryResponse:
    """
    Ask a natural-language question about an indexed repository.

    The question is answered through the existing RAG pipeline: retrieval
    over the indexed code, grounded prompt construction, and LLM generation.
    The response contains the answer plus the retrieved sources (file paths,
    line ranges, relevance). Raw embedding vectors are never returned.

    Repository semantics:
        - HTTP 404: the repository does not exist locally, OR it exists
          locally but has not been indexed yet (index it via
          POST /repository/index first).
        - HTTP 200 with ``insufficient_context=true``: the repository IS
          indexed, but no relevant context was found for the question, so
          no fabricated repository-specific answer is produced.
          ``insufficient_context`` is never conflated with a repository
          that does not exist.

    Args:
        request: Query request with repository name, question, and optional
            top_k.

    Returns:
        RagQueryResponse with the grounded answer and source provenance.

    Raises:
        HTTPException: 404 if the repository is not present locally or has
            not been indexed, 422 if the request body fails validation
            (empty question, empty repository name, invalid top_k), 500 for
            any retrieval or generation failure.
    """
    logger.info(
        f"Received RAG query for repository '{request.repository_name}': "
        f"{request.question!r}"
    )

    # Repository semantics (existing abstractions only): 404 when the
    # repository does not exist locally, or exists but was never indexed.
    # The message intentionally covers both cases. A failing existence/count
    # check is an unexpected server-side failure, surfaced as HTTP 500, so it
    # is never mistaken for "no relevant context".
    try:
        repository_exists = container.repository_manager.repository_exists(
            request.repository_name
        )
        repository_vectors = container.vector_store.count_by_repository(
            request.repository_name
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Repository state check failed for "
            f"'{request.repository_name}': {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the question.",
        )

    if not repository_exists or repository_vectors == 0:
        logger.info(
            f"Repository '{request.repository_name}' not queryable "
            f"(exists locally: {repository_exists})"
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Repository '{request.repository_name}' was not found or "
                "has not been indexed. Clone and index it first via "
                "POST /repository/clone and POST /repository/index."
            ),
        )

    try:
        rag_response = container.rag_service.answer(
            question=request.question,
            repository_name=request.repository_name,
            top_k=request.top_k,
        )
    except ValueError as e:
        logger.warning(f"Validation error in RAG query: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"RAG pipeline error for repository '{request.repository_name}': {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the question.",
        )

    sources = [
        RagQuerySource(
            file_path=result.relative_path,
            language=result.language,
            start_line=result.start_line,
            end_line=result.end_line,
            relevance_score=result.relevance_score,
        )
        for result in rag_response.retrieved_results
    ]

    response = RagQueryResponse(
        answer=rag_response.answer,
        repository_name=rag_response.repository_name,
        question=rag_response.question,
        sources=sources,
        insufficient_context=rag_response.insufficient_context,
        model_name=rag_response.model_name,
        provider_name=rag_response.provider_name,
        retrieved_count=rag_response.retrieved_count,
        context_included_count=rag_response.context_included_count,
    )

    logger.info(
        f"RAG query complete for '{response.repository_name}': "
        f"{len(sources)} sources, insufficient_context={response.insufficient_context}"
    )
    return response
