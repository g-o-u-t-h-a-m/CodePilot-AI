"""JavaScript code chunking strategy."""

import hashlib
import logging
import re
from typing import List, Optional

from app.chunking.strategy import ChunkStrategy
from app.chunking.models import CodeChunk, ChunkType
from app.models.source_file import SourceFile

logger = logging.getLogger(__name__)


class JavaScriptChunkStrategy(ChunkStrategy):
    """Chunking strategy for JavaScript and TypeScript files.

    Uses regex patterns to extract function and class definitions.
    Falls back to entire file as one chunk if no definitions found.
    """

    # Pattern to match function declarations
    FUNCTION_PATTERN = re.compile(
        r'^(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(',
        re.MULTILINE
    )

    # Pattern to match arrow functions assigned to const/let/var
    ARROW_FUNCTION_PATTERN = re.compile(
        r'^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(',
        re.MULTILINE
    )

    # Pattern to match class declarations
    CLASS_PATTERN = re.compile(
        r'^(?:export\s+)?class\s+(\w+)',
        re.MULTILINE
    )

    def chunk(self, source_file: SourceFile) -> List[CodeChunk]:
        """Split JavaScript/TypeScript source file into chunks.

        Args:
            source_file: The source file to chunk

        Returns:
            List of CodeChunk objects
        """
        chunks = []
        lines = source_file.content.splitlines()

        # Find all function and class definitions
        definitions = self._find_definitions(source_file.content, lines)

        if not definitions:
            logger.debug(
                f"No functions or classes found in {source_file.relative_path}, "
                "returning entire file as one chunk"
            )
            return [self._create_module_chunk(source_file)]

        # Sort definitions by line number
        definitions.sort(key=lambda x: x['start_line'])

        # Create chunks from definitions
        for idx, definition in enumerate(definitions):
            chunk = self._create_chunk_from_definition(
                definition, source_file, lines, idx
            )
            chunks.append(chunk)

        logger.debug(
            f"Created {len(chunks)} JavaScript chunks for {source_file.relative_path}"
        )

        return chunks

    def _find_definitions(self, content: str, lines: List[str]) -> List[dict]:
        """Find function and class definitions in the source.

        Args:
            content: Full source content
            lines: Lines of the source file

        Returns:
            List of definition dictionaries
        """
        definitions = []

        # Find regular functions
        for match in self.FUNCTION_PATTERN.finditer(content):
            start_pos = match.start()
            line_num = content[:start_pos].count('\n') + 1
            end_line = self._find_block_end(lines, line_num - 1)

            definitions.append({
                'type': ChunkType.FUNCTION,
                'name': match.group(1),
                'start_line': line_num,
                'end_line': end_line
            })

        # Find arrow functions
        for match in self.ARROW_FUNCTION_PATTERN.finditer(content):
            start_pos = match.start()
            line_num = content[:start_pos].count('\n') + 1
            end_line = self._find_block_end(lines, line_num - 1)

            definitions.append({
                'type': ChunkType.FUNCTION,
                'name': match.group(1),
                'start_line': line_num,
                'end_line': end_line
            })

        # Find classes
        for match in self.CLASS_PATTERN.finditer(content):
            start_pos = match.start()
            line_num = content[:start_pos].count('\n') + 1
            end_line = self._find_block_end(lines, line_num - 1)

            definitions.append({
                'type': ChunkType.CLASS,
                'name': match.group(1),
                'start_line': line_num,
                'end_line': end_line
            })

        return definitions

    def _find_block_end(self, lines: List[str], start_idx: int) -> int:
        """Find the end of a code block by tracking braces.

        Args:
            lines: Lines of the source file
            start_idx: Starting line index (0-based)

        Returns:
            Ending line number (1-based)
        """
        brace_count = 0
        found_opening = False

        for i in range(start_idx, len(lines)):
            line = lines[i]

            # Count braces
            for char in line:
                if char == '{':
                    brace_count += 1
                    found_opening = True
                elif char == '}':
                    brace_count -= 1

            # If we've found and closed all braces, this is the end
            if found_opening and brace_count == 0:
                return i + 1  # 1-based line number

        # If we didn't find proper closure, return last line
        return len(lines)

    def _create_chunk_from_definition(
        self,
        definition: dict,
        source_file: SourceFile,
        lines: List[str],
        index: int
    ) -> CodeChunk:
        """Create a chunk from a definition.

        Args:
            definition: Definition dictionary
            source_file: Source file
            lines: Lines of the source file
            index: Chunk index

        Returns:
            CodeChunk object
        """
        start_line = definition['start_line']
        end_line = definition['end_line']

        content = "\n".join(lines[start_line - 1:end_line])
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        metadata = {}
        if definition['type'] == ChunkType.FUNCTION:
            metadata['function_name'] = definition['name']
        elif definition['type'] == ChunkType.CLASS:
            metadata['class_name'] = definition['name']

        return CodeChunk(
            repository_name=self._extract_repo_name(source_file.path),
            relative_path=source_file.relative_path,
            language=source_file.language,
            chunk_type=definition['type'],
            chunk_index=index,
            start_line=start_line,
            end_line=end_line,
            content=content,
            content_hash=content_hash,
            metadata=metadata
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
        parts = path.replace("\\", "/").split("/")
        for i, part in enumerate(parts):
            if part == "indexed_repos" and i + 1 < len(parts):
                return parts[i + 1]
        return "unknown"
