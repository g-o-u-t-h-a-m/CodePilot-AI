"""BGE embedding provider implementation.

This module implements an embedding provider using the BGE (BAAI General Embedding)
model via the sentence-transformers library.

BGE models are optimized for semantic search and are well-suited for code embeddings.
"""

import logging
from typing import Optional

from sentence_transformers import SentenceTransformer

from app.chunking.models import CodeChunk
from app.embeddings.models import EmbeddingRecord
from app.embeddings.provider import EmbeddingProvider

logger = logging.getLogger(__name__)


class BGEProvider(EmbeddingProvider):
    """Embedding provider using BAAI/bge-small-en-v1.5 model.

    This provider:
    - Loads the BGE model once during initialization
    - Generates 384-dimensional embeddings
    - Normalizes embeddings for cosine similarity
    - Supports batch processing for efficiency

    The model is cached in memory and reused across multiple embed() calls.
    """

    MODEL_NAME = "BAAI/bge-small-en-v1.5"
    DIMENSION = 384

    def __init__(self):
        """Initialize the BGE provider and load the model."""
        logger.info(f"Initializing BGE provider with model: {self.MODEL_NAME}")
        self._model: Optional[SentenceTransformer] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the sentence transformer model.

        The model is loaded once and cached for subsequent calls.

        Raises:
            RuntimeError: If model loading fails
        """
        try:
            logger.info("Loading BGE model...")
            self._model = SentenceTransformer(self.MODEL_NAME)
            logger.info("BGE model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load BGE model: {e}")
            raise RuntimeError(f"Failed to load BGE model: {e}") from e

    def embed(self, chunk: CodeChunk) -> EmbeddingRecord:
        """Generate an embedding for a code chunk.

        Args:
            chunk: The code chunk to embed

        Returns:
            EmbeddingRecord containing the embedding and metadata

        Raises:
            RuntimeError: If embedding generation fails
        """
        if self._model is None:
            raise RuntimeError("BGE model not loaded")

        try:
            logger.debug(f"Generating embedding for chunk: {chunk.id}")

            # Generate embedding from chunk content
            # The encode method returns a numpy array
            embedding_array = self._model.encode(
                chunk.content,
                normalize_embeddings=True,  # Normalize for cosine similarity
                show_progress_bar=False
            )

            # Convert numpy array to list of floats
            embedding = embedding_array.tolist()

            # Create and return embedding record
            record = EmbeddingRecord(
                chunk_id=chunk.id,
                embedding=embedding,
                model_name=self.MODEL_NAME,
                dimension=self.DIMENSION,
                content_hash=chunk.content_hash
            )

            logger.debug(f"Embedding generated successfully for chunk: {chunk.id}")
            return record

        except Exception as e:
            logger.error(f"Failed to generate embedding for chunk {chunk.id}: {e}")
            raise RuntimeError(f"Failed to generate embedding: {e}") from e

    def get_model_name(self) -> str:
        """Get the name of the embedding model.

        Returns:
            Model name identifier
        """
        return self.MODEL_NAME

    def get_dimension(self) -> int:
        """Get the dimension of embeddings produced by this provider.

        Returns:
            Embedding vector dimension
        """
        return self.DIMENSION
