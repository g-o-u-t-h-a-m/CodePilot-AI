import logging
from typing import Optional
from app.models.repository import CloneRepositoryRequest, CloneRepositoryResponse
from app.repository.git_manager import RepositoryManager
from git import GitCommandError


logger = logging.getLogger(__name__)


class RepositoryService:
    """Service layer for repository operations."""

    def __init__(self, base_path: str = "indexed_repos"):
        """
        Initialize the RepositoryService.

        Args:
            base_path: Base directory where repositories will be stored
        """
        self.repository_manager = RepositoryManager(base_path=base_path)
        logger.info("RepositoryService initialized")

    def clone_repository(self, request: CloneRepositoryRequest) -> CloneRepositoryResponse:
        """
        Clone a repository from GitHub.

        Args:
            request: Clone repository request containing the GitHub URL

        Returns:
            CloneRepositoryResponse with operation result

        Raises:
            ValueError: If URL is invalid
            GitCommandError: If cloning fails
        """
        github_url = request.github_url.strip()

        logger.info(f"Processing clone request for URL: {github_url}")

        try:
            # Validate URL format
            if not self.repository_manager.validate_url(github_url):
                logger.warning(f"Invalid URL format: {github_url}")
                return CloneRepositoryResponse(
                    success=False,
                    repository_name="",
                    local_path="",
                    message=f"Invalid GitHub URL format: {github_url}"
                )

            # Extract repository name
            repo_name = self.repository_manager.extract_repo_name(github_url)

            # Check if repository already exists
            if self.repository_manager.repository_exists(repo_name):
                local_path = f"indexed_repos/{repo_name}/source"
                logger.info(f"Repository already exists: {repo_name}")
                return CloneRepositoryResponse(
                    success=True,
                    repository_name=repo_name,
                    local_path=local_path,
                    message="Repository already exists"
                )

            # Clone the repository
            repo_name, local_path = self.repository_manager.clone_repository(github_url)

            logger.info(f"Repository cloned successfully: {repo_name}")
            return CloneRepositoryResponse(
                success=True,
                repository_name=repo_name,
                local_path=local_path,
                message="Repository cloned successfully"
            )

        except ValueError as e:
            logger.error(f"Validation error: {e}")
            return CloneRepositoryResponse(
                success=False,
                repository_name="",
                local_path="",
                message=str(e)
            )

        except GitCommandError as e:
            logger.error(f"Git command error: {e}")
            return CloneRepositoryResponse(
                success=False,
                repository_name="",
                local_path="",
                message=f"Failed to clone repository: {str(e)}"
            )

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return CloneRepositoryResponse(
                success=False,
                repository_name="",
                local_path="",
                message=f"An unexpected error occurred: {str(e)}"
            )
