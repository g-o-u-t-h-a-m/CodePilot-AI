"""Python code chunking strategy using AST parsing."""

import ast
import hashlib
import logging
from typing import List

from app.chunking.strategy import ChunkStrategy
from app.chunking.models import CodeChunk, ChunkType
from app.models.source_file import SourceFile

logger = logging.getLogger(__name__)


class PythonChunkStrategy(ChunkStrategy):
    """Chunking strategy for Python source files.

    Uses Python's ast module to parse the code and extract functions
    and classes as individual chunks. If no functions or classes are found,
    the entire file is returned as a single chunk.
    """

    def chunk(self, source_file: SourceFile) -> List[CodeChunk]:
        """Split Python source file into chunks.

        Args:
            source_file: The Python source file to chunk

        Returns:
            List of CodeChunk objects
        """
        chunks = []

        try:
            tree = ast.parse(source_file.content)
            lines = source_file.content.splitlines()

            # Extract functions and classes
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    chunk = self._create_function_chunk(
                        node, source_file, lines, len(chunks)
                    )
                    if chunk:
                        chunks.append(chunk)
                elif isinstance(node, ast.ClassDef):
                    chunk = self._create_class_chunk(
                        node, source_file, lines, len(chunks)
                    )
                    if chunk:
                        chunks.append(chunk)

            # If no chunks found, return entire file as one chunk
            if not chunks:
                logger.debug(
                    f"No functions or classes found in {source_file.relative_path}, "
                    "returning entire file as one chunk"
                )
                chunks.append(self._create_module_chunk(source_file))

        except SyntaxError as e:
            logger.warning(
                f"Failed to parse {source_file.relative_path}: {e}. "
                "Returning entire file as one chunk"
            )
            chunks.append(self._create_module_chunk(source_file))

        return chunks

    def _create_function_chunk(
        self,
        node: ast.AST,
        source_file: SourceFile,
        lines: List[str],
        index: int
    ) -> CodeChunk:
        """Create a chunk for a function definition.

        Args:
            node: AST node representing the function
            source_file: Source file containing the function
            lines: Lines of the source file
            index: Chunk index

        Returns:
            CodeChunk for the function
        """
        start_line = node.lineno
        end_line = node.end_lineno or start_line

        content = "\n".join(lines[start_line - 1:end_line])
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        function_name = getattr(node, 'name', 'unknown')

        return CodeChunk(
            repository_name=self._extract_repo_name(source_file.path),
            relative_path=source_file.relative_path,
            language=source_file.language,
            chunk_type=ChunkType.FUNCTION,
            chunk_index=index,
            start_line=start_line,
            end_line=end_line,
            content=content,
            content_hash=content_hash,
            metadata={"function_name": function_name}
        )

    def _create_class_chunk(
        self,
        node: ast.ClassDef,
        source_file: SourceFile,
        lines: List[str],
        index: int
    ) -> CodeChunk:
        """Create a chunk for a class definition.

        Args:
            node: AST node representing the class
            source_file: Source file containing the class
            lines: Lines of the source file
            index: Chunk index

        Returns:
            CodeChunk for the class
        """
        start_line = node.lineno
        end_line = node.end_lineno or start_line

        content = "\n".join(lines[start_line - 1:end_line])
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        return CodeChunk(
            repository_name=self._extract_repo_name(source_file.path),
            relative_path=source_file.relative_path,
            language=source_file.language,
            chunk_type=ChunkType.CLASS,
            chunk_index=index,
            start_line=start_line,
            end_line=end_line,
            content=content,
            content_hash=content_hash,
            metadata={"class_name": node.name}
        )

    def _create_module_chunk(self, source_file: SourceFile) -> CodeChunk:
        """Create a chunk for an entire module.

        Args:
            source_file: Source file to chunk

        Returns:
            CodeChunk for the entire module
        """
        content_hash = hashlib.sha256(source_file.content.encode()).hexdigest()

        return CodeChunk(
            repository_name=self._extract_repo_name(source_file.path),
            relative_path=source_file.relative_path,
            language=source_file.language,
            chunk_type=ChunkType.MODULE,
            chunk_index=0,
            start_line=1,
            end_line=source_file.line_count,
            content=source_file.content,
            content_hash=content_hash,
            metadata={}
        )

    def _extract_repo_name(self, path: str) -> str:
        """Extract repository name from file path.

        Args:
            path: File path

        Returns:
            Repository name
        """
        # Extract from path like: /path/to/indexed_repos/repo-name/...
        parts = path.replace("\\", "/").split("/")
        for i, part in enumerate(parts):
            if part == "indexed_repos" and i + 1 < len(parts):
                return parts[i + 1]
        return "unknown"
