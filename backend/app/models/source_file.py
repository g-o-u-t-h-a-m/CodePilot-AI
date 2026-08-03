from pydantic import BaseModel, Field
from typing import Optional


class SourceFile(BaseModel):
    """Model representing a source code file scanned from a repository."""

    path: str = Field(
        ...,
        description="Absolute path to the file"
    )
    relative_path: str = Field(
        ...,
        description="Path relative to repository root"
    )
    extension: str = Field(
        ...,
        description="File extension (e.g., '.py', '.js')"
    )
    language: str = Field(
        ...,
        description="Programming language detected from extension"
    )
    size: int = Field(
        ...,
        description="File size in bytes"
    )
    line_count: int = Field(
        ...,
        description="Number of lines in the file"
    )
    encoding: str = Field(
        ...,
        description="Character encoding of the file (e.g., 'utf-8')"
    )
    sha256: str = Field(
        ...,
        description="SHA-256 hash of the file content"
    )
    content: str = Field(
        ...,
        description="Text content of the file"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "path": "/home/user/repo/src/main.py",
                "relative_path": "src/main.py",
                "extension": ".py",
                "language": "Python",
                "size": 1024,
                "line_count": 45,
                "encoding": "utf-8",
                "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "content": "# Sample Python code\nprint('Hello, World!')"
            }
        }
