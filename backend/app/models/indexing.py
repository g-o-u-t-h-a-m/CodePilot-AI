"""Request/response models for repository indexing.

The index endpoint reuses the repository layout established by the clone
flow (``indexed_repos/<name>/source``) and returns structured statistics
so clients can confirm how many files/chunks/embeddings/vectors were
produced by a single indexing operation.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class IndexRepositoryRequest(BaseModel):
    """Request model for indexing an already-cloned repository.

    Attributes:
        repository_name: Name of a repository that has already been cloned
            locally. This endpoint never clones from GitHub.
    """

    repository_name: str = Field(
        ...,
        description="Name of an already-cloned repository to index",
        examples=["demo-repository"],
        max_length=200,
    )

    @field_validator("repository_name")
    @classmethod
    def validate_repository_name(cls, value: str) -> str:
        """Reject empty/whitespace-only repository names."""
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("repository_name must not be empty or whitespace")
        return trimmed

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "repository_name": "demo-repository",
            }
        }
    )


class IndexRepositoryResponse(BaseModel):
    """Response model for a repository indexing operation.

    Attributes:
        success: Whether indexing completed successfully.
        repository_name: Name of the repository that was indexed.
        files_scanned: Number of source files found by the scanner.
        chunks_generated: Number of code chunks produced by the chunk engine.
        embeddings_generated: Number of embeddings produced by the engine.
        vectors_stored: Number of vectors persisted to the vector store.
        collection_count: Total vectors in the collection after this
            operation, across ALL repositories.
        repository_vectors: Vectors in the collection for THIS repository
            after this operation. Since re-indexing replaces the
            repository's previous vectors with the current state, this
            equals ``chunks_generated`` when indexing succeeds.
        duration: Indexing wall-clock duration in seconds.
        embedding_model: Name of the embedding model used (when known).
        message: Human-readable status message.
    """

    success: bool = Field(
        ...,
        description="Whether indexing completed successfully",
    )
    repository_name: str = Field(
        ...,
        description="Name of the repository that was indexed",
    )
    files_scanned: int = Field(
        ...,
        description="Number of source files scanned",
        ge=0,
    )
    chunks_generated: int = Field(
        ...,
        description="Number of code chunks generated",
        ge=0,
    )
    embeddings_generated: int = Field(
        ...,
        description="Number of embeddings generated",
        ge=0,
    )
    vectors_stored: int = Field(
        ...,
        description="Number of vectors stored in the vector store",
        ge=0,
    )
    collection_count: int = Field(
        ...,
        description=(
            "Total number of vectors in the collection (all repositories) "
            "after indexing"
        ),
        ge=0,
    )
    repository_vectors: int = Field(
        ...,
        description=(
            "Number of vectors in the collection for the indexed repository "
            "after indexing; equals chunks_generated after a successful "
            "replacement"
        ),
        ge=0,
    )
    duration: float = Field(
        ...,
        description="Indexing duration in seconds",
        ge=0.0,
    )
    embedding_model: Optional[str] = Field(
        default=None,
        description="Name of the embedding model used, if known",
    )
    message: str = Field(
        ...,
        description="Human-readable status message",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "repository_name": "demo-repository",
                "files_scanned": 4,
                "chunks_generated": 5,
                "embeddings_generated": 5,
                "vectors_stored": 5,
                "collection_count": 5,
                "repository_vectors": 5,
                "duration": 1.234,
                "embedding_model": "BAAI/bge-small-en-v1.5",
                "message": "Successfully indexed repository 'demo-repository'",
            }
        }
    )
