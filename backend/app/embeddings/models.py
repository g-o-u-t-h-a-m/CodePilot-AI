"""Data models for embeddings."""

from datetime import datetime
from typing import List
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class EmbeddingRecord(BaseModel):
    """Model representing an embedding generated from a code chunk.

    An embedding record stores the vector representation of a code chunk
    along with metadata for tracking and caching purposes.
    """

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the embedding record"
    )
    chunk_id: str = Field(
        ...,
        description="ID of the source code chunk"
    )
    embedding: List[float] = Field(
        ...,
        description="Vector embedding of the chunk content"
    )
    model_name: str = Field(
        ...,
        description="Name of the embedding model used"
    )
    dimension: int = Field(
        ...,
        description="Dimensionality of the embedding vector",
        ge=1
    )
    content_hash: str = Field(
        ...,
        description="Hash of the chunk content for cache lookup"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the embedding was created"
    )

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "chunk_id": "660e8400-e29b-41d4-a716-446655440001",
                "embedding": [0.123, -0.456, 0.789],
                "model_name": "BAAI/bge-small-en-v1.5",
                "dimension": 384,
                "content_hash": "abc123def456",
                "created_at": "2024-01-01T00:00:00"
            }
        }
    )
