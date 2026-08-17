import logging
from fastapi import APIRouter, HTTPException, status
from app.models.repository import CloneRepositoryRequest, CloneRepositoryResponse
from app.models.indexing import IndexRepositoryRequest, IndexRepositoryResponse
from app.services.container import container
from app.services.indexing_service import (
    IndexingError,
    RepositoryEmptyError,
    RepositoryMissingError,
    RepositoryUnavailableError,
)
from app.services.repository_service import RepositoryService


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/repository", tags=["repository"])

# Initialize service
repository_service = RepositoryService()


@router.post("/clone", response_model=CloneRepositoryResponse)
async def clone_repository(request: CloneRepositoryRequest) -> CloneRepositoryResponse:
    """
    Clone a GitHub repository.

    Args:
        request: Clone repository request containing the GitHub URL

    Returns:
        CloneRepositoryResponse with operation result

    Raises:
        HTTPException: If an unexpected error occurs
    """
    logger.info(f"Received clone request for URL: {request.github_url}")

    try:
        response = repository_service.clone_repository(request)

        if response.success:
            logger.info(f"Clone operation successful: {response.repository_name}")
        else:
            logger.warning(f"Clone operation failed: {response.message}")

        return response

    except Exception as e:
        logger.error(f"Unexpected error in clone endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.post(
    "/index",
    response_model=IndexRepositoryResponse,
    responses={
        status.HTTP_404_NOT_FOUND: {
            "description": "Repository is not cloned locally; clone it first"
        },
        status.HTTP_409_CONFLICT: {
            "description": "Repository directory exists but is not a valid clone"
        },
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "description": "Repository contains no supported source files"
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Indexing pipeline failure"
        },
    },
)
async def index_repository(
    request: IndexRepositoryRequest,
) -> IndexRepositoryResponse:
    """
    Index an already-cloned repository.

    This endpoint scans, chunks, embeds, and stores a repository that was
    cloned through ``POST /repository/clone``. It never clones from GitHub
    itself; if the repository is not present locally it returns HTTP 404.

    The business logic is delegated entirely to the IndexingService; this
    route only validates the request, maps expected failures to HTTP status
    codes, and logs the operation.

    Args:
        request: Index request containing the repository name.

    Returns:
        IndexRepositoryResponse with structured indexing statistics.

    Raises:
        HTTPException: 404 if the repository is not cloned locally,
            422 if the repository has no supported files,
            409 if the directory exists but is not a valid clone,
            500 for any unexpected pipeline failure.
    """
    logger.info(f"Received index request for repository: {request.repository_name}")

    try:
        response = container.indexing_service.index_repository(request)

        if response.success:
            logger.info(
                f"Indexing successful for '{response.repository_name}': "
                f"{response.files_scanned} files, "
                f"{response.chunks_generated} chunks, "
                f"{response.embeddings_generated} embeddings, "
                f"{response.vectors_stored} vectors "
                f"({response.duration}s)"
            )
        else:
            logger.warning(f"Indexing failed: {response.message}")

        return response

    except RepositoryMissingError as e:
        logger.warning(f"Repository not found for indexing: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except RepositoryUnavailableError as e:
        logger.warning(f"Repository unavailable for indexing: {e}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )
    except RepositoryEmptyError as e:
        logger.warning(f"Repository has no supported files: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except IndexingError as e:
        logger.error(f"Indexing pipeline error for '{request.repository_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Unexpected error in index endpoint for '{request.repository_name}': {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal error occurred while indexing the repository.",
        )
