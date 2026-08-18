"""Command-line client for the CodePilot backend (Sprint 10A).

The CLI is a pure HTTP client: it never imports or instantiates the core
CodePilot pipeline (RepositoryManager, RepositoryScanner, ChunkEngine,
EmbeddingEngine, ChromaDB, Retriever, PromptBuilder, RAGService,
LLMProvider). All operations are delegated to the existing FastAPI backend.

Modules:
    client: ``CodePilotClient`` HTTP abstraction.
    main:   argparse command layer (``python -m app.cli ...``).
"""

from app.cli.client import (
    CodePilotClient,
    CodePilotConnectionError,
    CodePilotError,
    CodePilotHTTPError,
    CodePilotProtocolError,
)

__all__ = [
    "CodePilotClient",
    "CodePilotError",
    "CodePilotConnectionError",
    "CodePilotHTTPError",
    "CodePilotProtocolError",
]
