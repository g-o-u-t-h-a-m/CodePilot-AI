import os
import re
import logging
from pathlib import Path
from typing import Tuple
from git import Repo, GitCommandError


logger = logging.getLogger(__name__)


class RepositoryManager:
    """Manages Git repository operations including cloning and validation."""

    def __init__(self, base_path: str = "indexed_repos"):
        """
        Initialize the RepositoryManager.

        Args:
            base_path: Base directory where repositories will be stored
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"RepositoryManager initialized with base_path: {self.base_path}")

    def validate_url(self, url: str) -> bool:
        """
        Validate if the URL is a valid GitHub repository URL.

        Args:
            url: GitHub repository URL to validate

        Returns:
            True if URL is valid, False otherwise
        """
        github_patterns = [
            r"^https?://github\.com/[\w\-]+/[\w\-\.]+/?$",
            r"^git@github\.com:[\w\-]+/[\w\-\.]+\.git$",
            r"^https?://github\.com/[\w\-]+/[\w\-\.]+\.git$"
        ]

        for pattern in github_patterns:
            if re.match(pattern, url):
                logger.info(f"URL validated successfully: {url}")
                return True

        logger.warning(f"Invalid GitHub URL format: {url}")
        return False

    def extract_repo_name(self, url: str) -> str:
        """
        Extract repository name from GitHub URL.

        Args:
            url: GitHub repository URL

        Returns:
            Repository name

        Raises:
            ValueError: If repository name cannot be extracted
        """
        # Remove .git suffix if present
        url = url.rstrip("/")
        if url.endswith(".git"):
            url = url[:-4]

        # Extract repo name from URL
        parts = url.split("/")
        if len(parts) >= 2:
            repo_name = parts[-1]
            logger.info(f"Extracted repository name: {repo_name}")
            return repo_name

        logger.error(f"Could not extract repository name from URL: {url}")
        raise ValueError(f"Invalid URL format: {url}")

    def repository_exists(self, repo_name: str) -> bool:
        """
        Check if repository already exists locally.

        Args:
            repo_name: Name of the repository

        Returns:
            True if repository exists, False otherwise
        """
        repo_path = self.base_path / repo_name / "source"
        exists = repo_path.exists() and (repo_path / ".git").exists()

        if exists:
            logger.info(f"Repository already exists: {repo_path}")
        else:
            logger.info(f"Repository does not exist: {repo_path}")

        return exists

    def clone_repository(self, url: str) -> Tuple[str, str]:
        """
        Clone a Git repository to the local filesystem.

        Args:
            url: GitHub repository URL to clone

        Returns:
            Tuple of (repository_name, local_path)

        Raises:
            ValueError: If URL is invalid
            GitCommandError: If cloning fails
        """
        # Validate URL
        if not self.validate_url(url):
            raise ValueError(f"Invalid GitHub URL: {url}")

        # Extract repository name
        repo_name = self.extract_repo_name(url)

        # Check if repository already exists
        if self.repository_exists(repo_name):
            local_path = str(self.base_path / repo_name / "source")
            logger.info(f"Repository already exists, returning existing path: {local_path}")
            return repo_name, local_path

        # Create destination path
        repo_path = self.base_path / repo_name / "source"
        repo_path.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(f"Cloning repository from {url} to {repo_path}")
            Repo.clone_from(url, str(repo_path))
            logger.info(f"Repository cloned successfully: {repo_path}")
            return repo_name, str(repo_path)

        except GitCommandError as e:
            logger.error(f"Failed to clone repository: {e}")
            # Clean up partial clone if it exists
            if repo_path.exists():
                import shutil
                shutil.rmtree(repo_path.parent)
                logger.info(f"Cleaned up partial clone at: {repo_path}")
            raise

        except Exception as e:
            logger.error(f"Unexpected error during cloning: {e}")
            raise
