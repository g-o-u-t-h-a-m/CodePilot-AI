"""Embedding provider implementations.

This package contains concrete implementations of embedding providers.
Each provider implements the EmbeddingProvider interface and can be
registered with the provider registry.

Available providers:
    - BGEProvider: Uses BAAI/bge-small-en-v1.5 model
"""

from app.embeddings.providers.bge import BGEProvider

__all__ = [
    "BGEProvider",
]
