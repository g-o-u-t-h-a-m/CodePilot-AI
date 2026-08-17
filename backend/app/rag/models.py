"""Data models for the retrieval (RAG) layer.

This module defines application-facing models for semantic retrieval,
independent of any specific vector database. The Retriever produces
RetrievalResult objects that represent a single relevant code chunk
found by querying the vector store with a natural-language question.

The model deliberately mirrors the metadata preserved by the vector
store (repository_name, relative_path, language, etc.) so that later
sprints (prompt building, LLM generation) can consume results without
knowing anything about ChromaDB or vector storage internals.
"""

from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.chunking.models import ChunkType


class RetrievalResult(BaseModel):
    """A single code chunk retrieved in response to a user question.

    This is the unit of retrieval: one relevant chunk, its source
    location, and a normalized relevance score. It is constructed by
    the Retriever from a vector store SimilarityResult, so it never
    exposes raw database distance semantics.

    Attributes:
        chunk_id: Identifier of the retrieved code chunk
        content: The chunk content (source code / text)
        repository_name: Repository the chunk belongs to
        relative_path: Path of the source file, relative to repo root
        language: Programming language of the chunk
        chunk_type: Type of chunk (function, class, module, ...)
        chunk_index: Index of the chunk within its source file (0-based)
        start_line: Starting line number of the chunk (1-based)
        end_line: Ending line number of the chunk (1-based)
        relevance_score: Normalized relevance in [0, 1]; higher = more relevant
        metadata: Full metadata preserved from the vector store
    """

    chunk_id: str = Field(
        ...,
        description="Identifier of the retrieved code chunk"
    )
    content: str = Field(
        ...,
        description="The chunk content (source code or text)"
    )
    repository_name: str = Field(
        ...,
        description="Repository the chunk belongs to"
    )
    relative_path: str = Field(
        ...,
        description="Path of the source file relative to repository root"
    )
    language: str = Field(
        ...,
        description="Programming language of the chunk"
    )
    chunk_type: ChunkType = Field(
        ...,
        description="Type of chunk (function, class, module, ...)"
    )
    chunk_index: int = Field(
        ...,
        description="Index of the chunk within its source file (0-based)",
        ge=0
    )
    start_line: int = Field(
        ...,
        description="Starting line number of the chunk (1-based)",
        ge=1
    )
    end_line: int = Field(
        ...,
        description="Ending line number of the chunk (1-based)",
        ge=1
    )
    relevance_score: float = Field(
        ...,
        description="Normalized relevance in [0, 1]; higher = more relevant",
        ge=0.0,
        le=1.0
    )
    metadata: Dict[str, object] = Field(
        default_factory=dict,
        description="Full metadata preserved from the vector store"
    )

    @field_validator("chunk_type", mode="before")
    @classmethod
    def coerce_chunk_type(cls, value) -> ChunkType:
        """Coerce a stored chunk type string into the ChunkType enum.

        ChromaDB persists the ChunkType as its string value (e.g.
        "function"), so this validator accepts both the enum member
        and its string form.

        Args:
            value: Raw chunk_type value (ChunkType or str)

        Returns:
            The validated ChunkType member

        Raises:
            ValueError: If the value is not a valid chunk type
        """
        if isinstance(value, ChunkType):
            return value
        if isinstance(value, str):
            return ChunkType(value)
        raise ValueError(
            f"Invalid chunk_type value: {value!r} "
            f"(expected ChunkType or str)"
        )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chunk_id": "550e8400-e29b-41d4-a716-446655440000",
                "content": "def authenticate_user():\n    pass",
                "repository_name": "demo-repo",
                "relative_path": "src/auth.py",
                "language": "Python",
                "chunk_type": "function",
                "chunk_index": 0,
                "start_line": 10,
                "end_line": 25,
                "relevance_score": 0.87,
                "metadata": {
                    "repository_name": "demo-repo",
                    "relative_path": "src/auth.py",
                    "language": "Python",
                    "chunk_type": "function",
                    "chunk_index": 0,
                    "start_line": 10,
                    "end_line": 25,
                    "content_hash": "abc123def456",
                    "model_name": "BAAI/bge-small-en-v1.5"
                }
            }
        }
    )


__all__ = ["RetrievalResult"]
