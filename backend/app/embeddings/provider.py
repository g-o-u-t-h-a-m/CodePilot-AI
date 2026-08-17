"""Abstract base class for embedding providers.

This module defines the interface that all embedding providers must implement.
Using the Strategy Pattern allows different embedding models to be used
interchangeably without modifying the embedding engine.
"""

from abc import ABC, abstractmethod

from app.chunking.models import CodeChunk
from app.embeddings.models import EmbeddingRecord


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    An embedding provider is responsible for:
    1. Loading and managing an embedding model
    2. Generating embeddings from code chunk content
    3. Creating EmbeddingRecord objects with proper metadata

    Implementations should:
    - Load the model once during initialization
    - Reuse the loaded model across multiple embed() calls
    - Handle model-specific preprocessing and normalization
    - Return embeddings in a consistent format
    """

    @abstractmethod
    def embed(self, chunk: CodeChunk) -> EmbeddingRecord:
        """Generate an embedding for a code chunk.

        Args:
            chunk: The code chunk to embed

        Returns:
            EmbeddingRecord containing the embedding and metadata

        Raises:
            RuntimeError: If embedding generation fails
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the name of the embedding model.

        Returns:
            Model name identifier
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimension of embeddings produced by this provider.

        Returns:
            Embedding vector dimension
        """
        pass
