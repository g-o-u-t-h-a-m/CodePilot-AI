"""Data models for code chunking."""

from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ChunkType(str, Enum):
    """Types of code chunks."""

    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    CONFIG = "config"
    DOCUMENT = "document"
    GENERIC = "generic"


class CodeChunk(BaseModel):
    """Model representing a chunk of code from a source file."""

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the chunk"
    )
    repository_name: str = Field(
        ...,
        description="Name of the repository this chunk belongs to"
    )
    relative_path: str = Field(
        ...,
        description="Path relative to repository root"
    )
    language: str = Field(
        ...,
        description="Programming language of the chunk"
    )
    chunk_type: ChunkType = Field(
        ...,
        description="Type of the chunk (function, class, module, etc.)"
    )
    chunk_index: int = Field(
        ...,
        description="Index of this chunk within the file (0-based)",
        ge=0
    )
    start_line: int = Field(
        ...,
        description="Starting line number (1-based)",
        ge=1
    )
    end_line: int = Field(
        ...,
        description="Ending line number (1-based)",
        ge=1
    )
    content: str = Field(
        ...,
        description="The actual code content of the chunk"
    )
    token_count: Optional[int] = Field(
        default=None,
        description="Estimated token count (optional)",
        ge=0
    )
    content_hash: str = Field(
        ...,
        description="Hash of the chunk content for deduplication"
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Additional metadata about the chunk"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "repository_name": "my-repo",
                "relative_path": "src/main.py",
                "language": "Python",
                "chunk_type": "function",
                "chunk_index": 0,
                "start_line": 10,
                "end_line": 25,
                "content": "def hello():\n    print('Hello, World!')",
                "token_count": 15,
                "content_hash": "abc123def456",
                "metadata": {"function_name": "hello"}
            }
        }
