"""JSON file chunking strategy."""

import hashlib
import logging
from typing import List

from app.chunking.strategy import ChunkStrategy
from app.chunking.models import CodeChunk, ChunkType
from app.models.source_file import SourceFile

logger = logging.getLogger(__name__)


class JsonChunkStrategy(ChunkStrategy):
    """Chunking strategy for JSON files.

    Treats the entire JSON file as a single chunk.
    """

    def chunk(self, source_file: SourceFile) -> List[CodeChunk]:
        """Chunk JSON file as a single unit.

        Args:
            source_file: The JSON source file to chunk

        Returns:
            List containing a single CodeChunk
        """
        content_hash = hashlib.sha256(source_file.content.encode()).hexdigest()

        chunk = CodeChunk(
            repository_name=self._extract_repo_name(source_file.path),
            relative_path=source_file.relative_path,
            language=source_file.language,
            chunk_type=ChunkType.CONFIG,
            chunk_index=0,
            start_line=1,
            end_line=source_file.line_count,
            content=source_file.content,
            content_hash=content_hash,
            metadata={"format": "json"}
        )

        logger.debug(f"Created single JSON chunk for {source_file.relative_path}")

        return [chunk]

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
