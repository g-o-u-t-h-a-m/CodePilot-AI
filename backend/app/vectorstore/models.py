"""Data models for the vector store layer.

These models are application-facing and independent of any specific
vector database implementation. They represent a stored record
(a chunk + its embedding + metadata) and a chunk/embedding pair
used for batch insertion.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.chunking.models import CodeChunk
from app.embeddings.models import EmbeddingRecord


class VectorStoreRecord(BaseModel):
    """A single record stored in the vector store.

    This is the application-facing representation of a stored
    chunk/embedding pair. It deliberately does not depend on any
    vector database response structures.

    Attributes:
        id: Unique identifier, matching the source CodeChunk.id
        embedding: The embedding vector (if included in the result)
        document: The chunk content
        metadata: Chunk and embedding metadata
    """

    id: str = Field(
        ...,
        description="Unique identifier, matching the source CodeChunk.id"
    )
    embedding: Optional[List[float]] = Field(
        default=None,
        description="The embedding vector, if requested"
    )
    document: Optional[str] = Field(
        default=None,
        description="The chunk content"
    )
    metadata: Dict[str, object] = Field(
        default_factory=dict,
        description="Chunk and embedding metadata"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "embedding": [0.123, -0.456, 0.789],
                "document": "def authenticate_user():\n    pass",
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


class ChunkEmbeddingPair(BaseModel):
    """A CodeChunk and its matching EmbeddingRecord.

    Used for batch insertion so the store can validate that the
    embedding record belongs to the chunk before persisting it.

    Attributes:
        chunk: The source code chunk
        record: The embedding record generated for the chunk
    """

    chunk: CodeChunk = Field(
        ...,
        description="The source code chunk"
    )
    record: EmbeddingRecord = Field(
        ...,
        description="The embedding record generated for the chunk"
    )
