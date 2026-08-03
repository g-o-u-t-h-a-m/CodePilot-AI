"""Generic fallback chunking strategy."""

import hashlib
import logging
from typing import List

from app.chunking.strategy import ChunkStrategy
from app.chunking.models import CodeChunk, ChunkType
from app.models.source_file import SourceFile

logger = logging.getLogger(__name__)


class GenericChunkStrategy(ChunkStrategy):
    """Fallback chunking strategy for unknown file types.

    Splits files into fixed-size chunks with overlap to preserve context
    across chunk boundaries.
    """

    def __init__(self, chunk_size: int = 120, overlap: int = 20):
        """Initialize the generic chunking strategy.

        Args:
            chunk_size: Number of lines per chunk
            overlap: Number of lines to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, source_file: SourceFile) -> List[CodeChunk]:
        """Split source file into fixed-size chunks with overlap.

        Args:
            source_file: The source file to chunk

        Returns:
            List of CodeChunk objects
        """
        chunks = []
        lines = source_file.content.splitlines()
        total_lines = len(lines)

        if total_lines == 0:
            logger.warning(f"Empty file: {source_file.relative_path}")
            return chunks

        chunk_index = 0
        start = 0

        while start < total_lines:
            end = min(start + self.chunk_size, total_lines)
            chunk_lines = lines[start:end]
            content = "\n".join(chunk_lines)

            content_hash = hashlib.sha256(content.encode()).hexdigest()

            chunks.append(CodeChunk(
                repository_name=self._extract_repo_name(source_file.path),
                relative_path=source_file.relative_path,
                language=source_file.language,
                chunk_type=ChunkType.GENERIC,
                chunk_index=chunk_index,
                start_line=start + 1,  # 1-based line numbers
                end_line=end,
                content=content,
                content_hash=content_hash,
                metadata={
                    "chunk_size": self.chunk_size,
                    "overlap": self.overlap
                }
            ))

            chunk_index += 1

            # Move to next chunk with overlap
            start = end - self.overlap if end < total_lines else end

        logger.debug(
            f"Created {len(chunks)} generic chunks for {source_file.relative_path}"
        )

        return chunks

    def _extract_repo_name(self, path: str) -> str:
        """Extract repository name from file path.

        Args:
            path: File path

        Returns:
            Repository name
        """
        parts = path.replace("\\", "/").split("/")
        for i, part in enumerate(parts):
            if part == "indexed_repos" and i + 1 < len(parts):
                return parts[i + 1]
        return "unknown"
