"""Markdown document chunking strategy."""

import hashlib
import logging
import re
from typing import List

from app.chunking.strategy import ChunkStrategy
from app.chunking.models import CodeChunk, ChunkType
from app.models.source_file import SourceFile

logger = logging.getLogger(__name__)


class MarkdownChunkStrategy(ChunkStrategy):
    """Chunking strategy for Markdown documents.

    Splits markdown files by headings, with each section becoming
    a separate chunk.
    """

    def chunk(self, source_file: SourceFile) -> List[CodeChunk]:
        """Split Markdown file into chunks by headings.

        Args:
            source_file: The Markdown source file to chunk

        Returns:
            List of CodeChunk objects
        """
        chunks = []
        lines = source_file.content.splitlines()

        # Find all heading positions
        heading_positions = []
        for i, line in enumerate(lines):
            if re.match(r'^#+\s+', line):
                heading_positions.append(i)

        # If no headings found, return entire file as one chunk
        if not heading_positions:
            logger.debug(
                f"No headings found in {source_file.relative_path}, "
                "returning entire file as one chunk"
            )
            return [self._create_document_chunk(source_file, 0, 1, len(lines), lines)]

        # Create chunks for each section
        for idx, start_pos in enumerate(heading_positions):
            # Determine end position
            if idx + 1 < len(heading_positions):
                end_pos = heading_positions[idx + 1] - 1
            else:
                end_pos = len(lines) - 1

            chunk = self._create_document_chunk(
                source_file,
                idx,
                start_pos + 1,  # 1-based line numbers
                end_pos + 1,
                lines[start_pos:end_pos + 1]
            )
            chunks.append(chunk)

        logger.debug(
            f"Created {len(chunks)} markdown chunks for {source_file.relative_path}"
        )

        return chunks

    def _create_document_chunk(
        self,
        source_file: SourceFile,
        index: int,
        start_line: int,
        end_line: int,
        chunk_lines: List[str]
    ) -> CodeChunk:
        """Create a document chunk.

        Args:
            source_file: Source file
            index: Chunk index
            start_line: Starting line number (1-based)
            end_line: Ending line number (1-based)
            chunk_lines: Lines of content

        Returns:
            CodeChunk for the document section
        """
        content = "\n".join(chunk_lines)
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Extract heading text if present
        heading = ""
        if chunk_lines and re.match(r'^#+\s+', chunk_lines[0]):
            heading = re.sub(r'^#+\s+', '', chunk_lines[0]).strip()

        return CodeChunk(
            repository_name=self._extract_repo_name(source_file.path),
            relative_path=source_file.relative_path,
            language=source_file.language,
            chunk_type=ChunkType.DOCUMENT,
            chunk_index=index,
            start_line=start_line,
            end_line=end_line,
            content=content,
            content_hash=content_hash,
            metadata={"heading": heading} if heading else {}
        )

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
