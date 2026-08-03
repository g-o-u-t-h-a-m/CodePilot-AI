"""Code chunking module.

This module provides a flexible, extensible architecture for chunking
source code files into meaningful segments using the Strategy Pattern
and Registry Pattern.

Main components:
    - ChunkEngine: Main entry point for chunking operations
    - ChunkStrategy: Abstract base class for chunking strategies
    - StrategyRegistry: Registry for managing and retrieving strategies
    - CodeChunk: Data model representing a code chunk
    - ChunkType: Enumeration of chunk types

Example usage:
    from app.chunking import ChunkEngine, initialize_strategies
    from app.models.source_file import SourceFile

    # Initialize the strategy registry
    initialize_strategies()

    # Create the chunk engine
    engine = ChunkEngine()

    # Chunk a source file
    chunks = engine.chunk_file(source_file)

    # Process chunks
    for chunk in chunks:
        print(f"{chunk.chunk_type}: {chunk.relative_path}:{chunk.start_line}")
"""

from app.chunking.engine import ChunkEngine
from app.chunking.models import CodeChunk, ChunkType
from app.chunking.strategy import ChunkStrategy
from app.chunking.registry import (
    get_registry,
    register_strategy,
    register_fallback_strategy,
)
from app.chunking.strategies import (
    PythonChunkStrategy,
    JavaScriptChunkStrategy,
    JavaChunkStrategy,
    GenericCodeChunkStrategy,
    MarkdownChunkStrategy,
    JsonChunkStrategy,
    YamlChunkStrategy,
    GenericChunkStrategy,
)

import logging

logger = logging.getLogger(__name__)


def initialize_strategies():
    """Initialize and register all chunking strategies.

    This function must be called once during application startup
    to populate the strategy registry.

    The registry maps languages and file extensions to appropriate
    chunking strategies. This approach follows the Open/Closed Principle:
    new strategies can be added without modifying the ChunkEngine.
    """
    logger.info("Initializing chunking strategies")

    # Register Python strategy
    register_strategy("python", PythonChunkStrategy)
    register_strategy(".py", PythonChunkStrategy)

    # Register JavaScript/TypeScript strategies
    register_strategy("javascript", JavaScriptChunkStrategy)
    register_strategy("typescript", JavaScriptChunkStrategy)
    register_strategy(".js", JavaScriptChunkStrategy)
    register_strategy(".jsx", JavaScriptChunkStrategy)
    register_strategy(".ts", JavaScriptChunkStrategy)
    register_strategy(".tsx", JavaScriptChunkStrategy)
    register_strategy(".mjs", JavaScriptChunkStrategy)
    register_strategy(".cjs", JavaScriptChunkStrategy)

    # Register Java strategy
    register_strategy("java", JavaChunkStrategy)
    register_strategy(".java", JavaChunkStrategy)

    # Register Markdown strategy
    register_strategy("markdown", MarkdownChunkStrategy)
    register_strategy(".md", MarkdownChunkStrategy)
    register_strategy(".markdown", MarkdownChunkStrategy)

    # Register JSON strategy
    register_strategy("json", JsonChunkStrategy)
    register_strategy(".json", JsonChunkStrategy)

    # Register YAML strategy
    register_strategy("yaml", YamlChunkStrategy)
    register_strategy(".yaml", YamlChunkStrategy)
    register_strategy(".yml", YamlChunkStrategy)

    # Register generic code strategy for common programming languages
    # that don't have dedicated parsers yet
    register_strategy("c", GenericCodeChunkStrategy)
    register_strategy(".c", GenericCodeChunkStrategy)
    register_strategy(".h", GenericCodeChunkStrategy)
    register_strategy("c++", GenericCodeChunkStrategy)
    register_strategy("cpp", GenericCodeChunkStrategy)
    register_strategy(".cpp", GenericCodeChunkStrategy)
    register_strategy(".hpp", GenericCodeChunkStrategy)
    register_strategy(".cc", GenericCodeChunkStrategy)
    register_strategy("c#", GenericCodeChunkStrategy)
    register_strategy("csharp", GenericCodeChunkStrategy)
    register_strategy(".cs", GenericCodeChunkStrategy)
    register_strategy("go", GenericCodeChunkStrategy)
    register_strategy(".go", GenericCodeChunkStrategy)
    register_strategy("rust", GenericCodeChunkStrategy)
    register_strategy(".rs", GenericCodeChunkStrategy)
    register_strategy("ruby", GenericCodeChunkStrategy)
    register_strategy(".rb", GenericCodeChunkStrategy)
    register_strategy("php", GenericCodeChunkStrategy)
    register_strategy(".php", GenericCodeChunkStrategy)
    register_strategy("swift", GenericCodeChunkStrategy)
    register_strategy(".swift", GenericCodeChunkStrategy)
    register_strategy("kotlin", GenericCodeChunkStrategy)
    register_strategy(".kt", GenericCodeChunkStrategy)

    # Register fallback strategy for unknown file types
    register_fallback_strategy(GenericChunkStrategy)

    logger.info("Chunking strategies initialized successfully")


__all__ = [
    "ChunkEngine",
    "CodeChunk",
    "ChunkType",
    "ChunkStrategy",
    "initialize_strategies",
    "get_registry",
    "register_strategy",
    "register_fallback_strategy",
]
