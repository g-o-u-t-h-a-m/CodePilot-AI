"""Abstract base class for chunking strategies."""

from abc import ABC, abstractmethod
from typing import List

from app.models.source_file import SourceFile
from app.chunking.models import CodeChunk


class ChunkStrategy(ABC):
    """Abstract base class for code chunking strategies.

    Each strategy implements language-specific or content-specific
    logic for splitting source files into meaningful chunks.
    """

    @abstractmethod
    def chunk(self, source_file: SourceFile) -> List[CodeChunk]:
        """Split a source file into chunks.

        Args:
            source_file: The source file to chunk

        Returns:
            List of CodeChunk objects
        """
        pass
