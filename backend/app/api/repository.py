import logging
from fastapi import APIRouter, HTTPException, status
from app.models.repository import CloneRepositoryRequest, CloneRepositoryResponse
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
