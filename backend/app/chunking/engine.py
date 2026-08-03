"""Core chunk engine for processing source files into chunks.

The ChunkEngine is the main entry point for chunking operations.
It uses the Strategy Pattern and Registry Pattern to delegate chunking
logic to language-specific or content-specific strategies.
"""

import logging
from typing import List

from app.chunking.models import CodeChunk
from app.chunking.registry import get_registry
from app.chunking.strategy import ChunkStrategy
from app.models.source_file import SourceFile

logger = logging.getLogger(__name__)


class ChunkEngine:
    """Main engine for chunking source files.

    The ChunkEngine is responsible for:
    1. Receiving a SourceFile
    2. Retrieving the appropriate chunking strategy from the registry
    3. Delegating chunking logic to the strategy
    4. Returning the generated CodeChunks

    The engine follows the Open/Closed Principle - it is closed for modification
    but open for extension through the strategy registry.
    """

    def __init__(self):
        """Initialize the chunk engine."""
        self.registry = get_registry()
        logger.info("ChunkEngine initialized")

    def chunk_file(self, source_file: SourceFile) -> List[CodeChunk]:
        """Chunk a source file into CodeChunks.

        This is the main entry point for chunking operations.

        Args:
            source_file: The source file to chunk

        Returns:
            List of CodeChunk objects

        Raises:
            ValueError: If no strategy can be found for the file
        """
        logger.info(f"Chunking file: {source_file.relative_path}")

        # Get the appropriate strategy from the registry
        strategy_class = self._get_strategy(source_file)

        # Instantiate the strategy
        strategy = strategy_class()

        # Delegate chunking to the strategy
        chunks = strategy.chunk(source_file)

        logger.info(
            f"Generated {len(chunks)} chunks for {source_file.relative_path} "
            f"using {strategy_class.__name__}"
        )

        return chunks

    def chunk_files(self, source_files: List[SourceFile]) -> List[CodeChunk]:
        """Chunk multiple source files.

        Args:
            source_files: List of source files to chunk

        Returns:
            Flat list of all CodeChunks from all files
        """
        all_chunks = []

        for source_file in source_files:
            try:
                chunks = self.chunk_file(source_file)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(
                    f"Failed to chunk file {source_file.relative_path}: {e}",
                    exc_info=True
                )
                # Continue processing other files
                continue

        logger.info(
            f"Generated {len(all_chunks)} total chunks from {len(source_files)} files"
        )

        return all_chunks

    def _get_strategy(self, source_file: SourceFile) -> type[ChunkStrategy]:
        """Get the appropriate chunking strategy for a source file.

        The strategy is selected based on:
        1. Language (e.g., 'python', 'javascript')
        2. File extension (e.g., '.md', '.json')
        3. Fallback to generic strategy if no match

        Args:
            source_file: The source file to get a strategy for

        Returns:
            Strategy class to use for chunking
        """
        # Try to get strategy by language first
        language_lower = source_file.language.lower()

        try:
            strategy = self.registry.get_strategy(language_lower)
            logger.debug(f"Using language-based strategy for '{language_lower}'")
            return strategy
        except ValueError:
            pass

        # Try to get strategy by file extension
        extension_lower = source_file.extension.lower()

        try:
            strategy = self.registry.get_strategy(extension_lower)
            logger.debug(f"Using extension-based strategy for '{extension_lower}'")
            return strategy
        except ValueError:
            pass

        # This should not happen if fallback is registered, but handle it
        logger.warning(
            f"No strategy found for {source_file.relative_path}, "
            "and no fallback available"
        )
        raise ValueError(
            f"No chunking strategy available for {source_file.language} "
            f"({source_file.extension})"
        )
