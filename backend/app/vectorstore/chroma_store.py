"""ChromaDB vector store implementation.

This module provides a concrete VectorStore implementation backed by
ChromaDB's persistent local storage. It stores chunk/embedding pairs
along with their metadata, and supports batch operations and upsert
semantics so that re-inserting an existing chunk ID does not create
duplicate records.

The store does NOT generate embeddings; it receives already-generated
EmbeddingRecords from the EmbeddingEngine and persists them as-is.
"""

import logging
import os
from typing import Dict, List, Optional

import numpy as np
from chromadb import PersistentClient
from chromadb.api.models.Collection import Collection

from app.chunking.models import CodeChunk
from app.embeddings.models import EmbeddingRecord
from app.vectorstore.models import ChunkEmbeddingPair, VectorStoreRecord
from app.vectorstore.store import VectorStore

logger = logging.getLogger(__name__)

# Default collection name for chunk embeddings
DEFAULT_COLLECTION_NAME = "code_chunks"

# Default persistence path relative to the backend directory
DEFAULT_CHROMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "chroma_db"
)

# Metadata keys copied from CodeChunk, mapped to ChromaDB-safe values
_METADATA_KEYS = [
    "repository_name",
    "relative_path",
    "language",
    "chunk_type",
    "chunk_index",
    "start_line",
    "end_line",
]


class ChromaVectorStore(VectorStore):
    """ChromaDB-backed implementation of the VectorStore interface.

    Uses ChromaDB's persistent local storage so that data survives
    process restarts. The persistence path and collection name are
    configurable at construction time.

    Design notes:
    - IDs are the CodeChunk.id, preserving the relationship
      CodeChunk.id -> EmbeddingRecord.chunk_id -> ChromaDB record ID.
    - Upsert is used for single and batch insertion so duplicate IDs
      overwrite rather than create duplicate records.
    - repository_name is stored in metadata for repository isolation
      and filtering in later sprints.
    """

    def __init__(
        self,
        persistence_path: Optional[str] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME
    ):
        """Initialize the ChromaDB vector store.

        Args:
            persistence_path: Directory for ChromaDB persistent storage.
                Defaults to backend/chroma_db.
            collection_name: Name of the ChromaDB collection to use.
                Defaults to "code_chunks".

        Raises:
            RuntimeError: If ChromaDB client or collection initialization fails
        """
        self._path = persistence_path or DEFAULT_CHROMA_PATH

        try:
            os.makedirs(self._path, exist_ok=True)
            self._client = PersistentClient(path=self._path)
            logger.info(
                f"ChromaDB client initialized at: {self._path}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise RuntimeError(f"Failed to initialize ChromaDB client: {e}") from e

        try:
            self._collection: Collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(
                f"ChromaDB collection '{collection_name}' ready "
                f"(count: {self._collection.count()})"
            )
        except Exception as e:
            logger.error(f"Failed to create ChromaDB collection: {e}")
            raise RuntimeError(f"Failed to create ChromaDB collection: {e}") from e

    # ------------------------------------------------------------------
    # Public VectorStore API
    # ------------------------------------------------------------------

    def add(self, chunk: CodeChunk, record: EmbeddingRecord) -> None:
        """Store a single chunk/embedding pair.

        Args:
            chunk: The source code chunk
            record: The embedding record generated for the chunk

        Raises:
            ValueError: If the embedding record does not belong to the chunk
            RuntimeError: If ChromaDB insertion fails
        """
        self._validate_pair(chunk, record)

        metadata = self._build_metadata(chunk, record)
        logger.info(
            f"Upserting chunk {chunk.id} into collection "
            f"(dimension: {record.dimension})"
        )

        try:
            self._collection.upsert(
                ids=[chunk.id],
                embeddings=[record.embedding],
                documents=[chunk.content],
                metadatas=[metadata]
            )
            logger.info(f"Upserted chunk {chunk.id}")
        except Exception as e:
            logger.error(f"Failed to upsert chunk {chunk.id}: {e}")
            raise RuntimeError(f"Failed to store chunk {chunk.id}: {e}") from e

    def add_many(self, pairs: List[ChunkEmbeddingPair]) -> int:
        """Store multiple chunk/embedding pairs in a single batch.

        Args:
            pairs: List of chunk/embedding pairs to store

        Returns:
            Number of pairs successfully stored

        Raises:
            RuntimeError: If the entire batch insertion fails
        """
        if not pairs:
            logger.info("add_many called with empty list; nothing to do")
            return 0

        ids: List[str] = []
        embeddings: List[List[float]] = []
        documents: List[str] = []
        metadatas: List[Dict[str, object]] = []

        for pair in pairs:
            chunk = pair.chunk
            record = pair.record

            try:
                self._validate_pair(chunk, record)
            except ValueError as e:
                logger.error(f"Skipping invalid pair in batch: {e}")
                continue

            ids.append(chunk.id)
            embeddings.append(record.embedding)
            documents.append(chunk.content)
            metadatas.append(self._build_metadata(chunk, record))

        if not ids:
            logger.warning("No valid pairs to upsert in batch")
            return 0

        logger.info(
            f"Batch upserting {len(ids)} chunks into collection"
        )

        try:
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"Batch upserted {len(ids)} chunks")
            return len(ids)
        except Exception as e:
            logger.error(f"Failed to batch upsert {len(ids)} chunks: {e}")
            raise RuntimeError(f"Failed to batch upsert chunks: {e}") from e

    def get(
        self,
        chunk_id: str,
        include_embedding: bool = True
    ) -> Optional[VectorStoreRecord]:
        """Retrieve a stored record by chunk ID.

        Args:
            chunk_id: The chunk ID to look up
            include_embedding: Whether to include the embedding vector

        Returns:
            The stored record, or None if not found

        Raises:
            RuntimeError: If ChromaDB retrieval fails
        """
        logger.info(f"Retrieving chunk {chunk_id}")

        include = ["metadatas", "documents"]
        if include_embedding:
            include.append("embeddings")

        try:
            result = self._collection.get(
                ids=[chunk_id],
                include=include
            )
        except Exception as e:
            logger.error(f"Failed to retrieve chunk {chunk_id}: {e}")
            raise RuntimeError(f"Failed to retrieve chunk {chunk_id}: {e}") from e

        ids = result.get("ids") or []
        if not ids:
            logger.info(f"Chunk {chunk_id} not found")
            return None

        record = self._to_record(result, index=0)
        logger.info(f"Retrieved chunk {chunk_id}")
        return record

    def delete(self, chunk_id: str) -> bool:
        """Delete a stored record by chunk ID.

        Args:
            chunk_id: The chunk ID to delete

        Returns:
            True if a record was deleted, False if it did not exist

        Raises:
            RuntimeError: If ChromaDB deletion fails
        """
        # Check existence first so we can report a meaningful result.
        existing = self.get(chunk_id, include_embedding=False)
        if existing is None:
            logger.info(f"Chunk {chunk_id} not found; nothing to delete")
            return False

        logger.info(f"Deleting chunk {chunk_id}")
        try:
            self._collection.delete(ids=[chunk_id])
            logger.info(f"Deleted chunk {chunk_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete chunk {chunk_id}: {e}")
            raise RuntimeError(f"Failed to delete chunk {chunk_id}: {e}") from e

    def count(self) -> int:
        """Count the number of stored records.

        Returns:
            Total number of records currently stored

        Raises:
            RuntimeError: If ChromaDB count fails
        """
        try:
            total = self._collection.count()
            logger.debug(f"Collection count: {total}")
            return total
        except Exception as e:
            logger.error(f"Failed to count collection: {e}")
            raise RuntimeError(f"Failed to count collection: {e}") from e

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_pair(self, chunk: CodeChunk, record: EmbeddingRecord) -> None:
        """Validate that an embedding record belongs to a code chunk.

        Args:
            chunk: The source code chunk
            record: The embedding record

        Raises:
            ValueError: If the record does not belong to the chunk
        """
        if record.chunk_id != chunk.id:
            raise ValueError(
                f"Embedding record chunk_id '{record.chunk_id}' does not "
                f"match CodeChunk id '{chunk.id}'"
            )

        if record.content_hash != chunk.content_hash:
            raise ValueError(
                f"Embedding record content_hash does not match CodeChunk "
                f"content_hash for chunk {chunk.id}"
            )

    def _build_metadata(
        self,
        chunk: CodeChunk,
        record: EmbeddingRecord
    ) -> Dict[str, object]:
        """Build ChromaDB-compatible metadata for a chunk/embedding pair.

        ChromaDB metadata values must be str, int, float, or bool. All
        selected fields are already ChromaDB-safe, so no coercion is
        needed beyond a defensive check.

        Args:
            chunk: The source code chunk
            record: The embedding record

        Returns:
            Metadata dictionary suitable for ChromaDB

        Raises:
            ValueError: If any metadata value is not ChromaDB-compatible
        """
        metadata: Dict[str, object] = {}

        for key in _METADATA_KEYS:
            value = getattr(chunk, key, None)
            if value is not None:
                if isinstance(value, bool) or isinstance(value, (str, int, float)):
                    metadata[key] = value
                else:
                    # chunk_type is a ChunkType enum; store its value
                    metadata[key] = str(value)

        metadata["content_hash"] = record.content_hash
        metadata["model_name"] = record.model_name

        # Defensive check: every value must be ChromaDB-compatible.
        for key, value in metadata.items():
            if not isinstance(value, bool) and not isinstance(
                value, (str, int, float)
            ):
                raise ValueError(
                    f"Metadata value for '{key}' is not ChromaDB-compatible "
                    f"({type(value).__name__})"
                )

        return metadata

    def _to_record(self, result: dict, index: int) -> VectorStoreRecord:
        """Convert a ChromaDB get() result entry into a VectorStoreRecord.

        Args:
            result: The ChromaDB get() result dictionary
            index: Index of the record within the result

        Returns:
            An application-facing VectorStoreRecord
        """
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        embeddings = result.get("embeddings")

        record_id = ids[index]
        metadata = dict(metadatas[index]) if index < len(metadatas) else {}

        document = None
        if index < len(documents) and documents[index] is not None:
            document = documents[index]

        embedding = None
        if embeddings is not None and index < len(embeddings):
            raw = embeddings[index]
            # ChromaDB returns numpy arrays; normalize to a plain list.
            embedding = np.asarray(raw).tolist()

        return VectorStoreRecord(
            id=record_id,
            embedding=embedding,
            document=document,
            metadata=metadata
        )
