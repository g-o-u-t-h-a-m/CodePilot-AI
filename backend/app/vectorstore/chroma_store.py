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
from app.vectorstore.models import (
    ChunkEmbeddingPair,
    SimilarityResult,
    VectorStoreRecord,
)
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

# Max records per upsert call. ChromaDB rejects single requests larger than
# its max_batch_size (5461 for the local persistent client); batching well
# below that keeps one large repository from failing the whole upsert.
_CHROMA_UPSERT_BATCH_SIZE = 2000


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

        self._upsert_batches(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info(f"Batch upserted {len(ids)} chunks")
        return len(ids)

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

    def delete_by_repository(self, repository_name: str) -> int:
        """Delete every stored record that belongs to a repository.

        ChromaDB records carry ``repository_name`` in their metadata, so the
        repository's records are found with an exact-match ``where`` filter
        and removed in one batched delete. This lets re-indexing replace a
        repository's vectors wholesale; repositories with no stored records
        simply delete zero records.

        Args:
            repository_name: The repository whose records should be deleted.

        Returns:
            Number of records deleted. Zero if the repository has no stored
            records (which is not an error).

        Raises:
            RuntimeError: If ChromaDB deletion fails
        """
        logger.info(f"Fetching existing records for repository '{repository_name}'")
        existing = self._collection.get(
            where={"repository_name": repository_name},
            include=["metadatas"],
        )
        ids = existing.get("ids") or []
        if not ids:
            logger.info(
                f"Repository '{repository_name}' has no existing records; "
                "nothing to delete"
            )
            return 0

        logger.info(
            f"Deleting {len(ids)} existing records for repository "
            f"'{repository_name}'"
        )
        try:
            self._collection.delete(ids=ids)
            logger.info(
                f"Deleted {len(ids)} records for repository '{repository_name}'"
            )
            return len(ids)
        except Exception as e:
            logger.error(
                f"Failed to delete records for repository '{repository_name}': {e}"
            )
            raise RuntimeError(
                f"Failed to delete records for repository "
                f"'{repository_name}': {e}"
            ) from e

    def count_by_repository(self, repository_name: str) -> int:
        """Count the number of stored records for a repository.

        The collection's ``repository_name`` metadata is filtered with an
        exact-match ``where``; the count is derived from the matching IDs.

        Args:
            repository_name: The repository to count records for.

        Returns:
            Number of records currently stored for the repository.

        Raises:
            RuntimeError: If ChromaDB retrieval of the filtered IDs fails
        """
        logger.info(f"Counting records for repository '{repository_name}'")
        try:
            records = self._collection.get(
                where={"repository_name": repository_name},
                include=["metadatas"],
            )
            total = len(records.get("ids") or [])
            logger.info(
                f"Repository '{repository_name}' has {total} records"
            )
            return total
        except Exception as e:
            logger.error(
                f"Failed to count records for repository '{repository_name}': {e}"
            )
            raise RuntimeError(
                f"Failed to count records for repository "
                f"'{repository_name}': {e}"
            ) from e

    def query_similar(
        self,
        embedding: List[float],
        top_k: int,
        filter_metadata: Optional[Dict[str, object]] = None
    ) -> List[SimilarityResult]:
        """Query the store for records most similar to a query embedding.

        Uses ChromaDB's cosine metric (the collection is created with
        "hnsw:space": "cosine"). ChromaDB returns a cosine *distance* in
        [0, 2] where a lower value means more similar. We convert it to a
        normalized similarity score as: score = 1 - distance.

        This conversion is mathematically valid for the cosine metric:
        cosine distance is defined as (1 - cosine_similarity), so
        (1 - distance) recovers cosine similarity exactly, which lies in
        [-1, 1] for the raw vectors and in [0, 1] here because the BGE
        provider normalizes embeddings. The application-facing score is
        therefore "higher = more relevant", consistent with the cosine
        metric, and raw ChromaDB distance is never exposed.

        Results are returned ordered from most relevant to least relevant
        (ChromaDB returns query results in that order).

        Args:
            embedding: The query embedding vector
            top_k: Maximum number of results to return
            filter_metadata: Optional metadata filter (e.g. repository_name)
                applied as an exact-match filter before ranking

        Returns:
            List of SimilarityResult ordered from most to least relevant.
            Empty list if no records match.

        Raises:
            ValueError: If embedding is empty or top_k is not positive
            RuntimeError: If ChromaDB query fails
        """
        if not embedding:
            raise ValueError("Query embedding must not be empty")
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        where = None
        if filter_metadata:
            # All ChromaDB metadata filters are ANDed together, so a
            # query with multiple conditions becomes a dict of equality
            # constraints.
            where = {
                key: value
                for key, value in filter_metadata.items()
                if key and value is not None
            }

        logger.info(
            f"Querying {top_k} most similar records "
            f"(dimension: {len(embedding)}, filter: {where or 'none'})"
        )

        try:
            result = self._collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                where=where,
                include=["metadatas", "documents", "distances"]
            )
        except Exception as e:
            logger.error(f"Failed to query similar records: {e}")
            raise RuntimeError(f"Failed to query similar records: {e}") from e

        # ChromaDB returns lists-of-lists (one entry per query embedding);
        # we queried with a single embedding.
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]

        if not ids:
            logger.info("No similar records found")
            return []

        results: List[SimilarityResult] = []
        for index, record_id in enumerate(ids):
            distance = distances[index] if index < len(distances) else 1.0
            score = max(0.0, min(1.0, 1.0 - float(distance)))

            metadata = {}
            if index < len(metadatas) and metadatas[index] is not None:
                metadata = dict(metadatas[index])

            document = None
            if index < len(documents) and documents[index] is not None:
                document = documents[index]

            results.append(
                SimilarityResult(
                    id=record_id,
                    score=score,
                    document=document,
                    metadata=metadata
                )
            )

        logger.info(f"Query returned {len(results)} results")
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _upsert_batches(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, object]],
    ) -> None:
        """Upsert records in batches that stay below ChromaDB's limit.

        ChromaDB rejects a single ``upsert`` whose record count exceeds its
        ``max_batch_size`` (5461 for the local persistent client). Large
        repositories can produce more chunks than that in one indexing pass
        (LoyalBasket produces 6017), so the record lists are split into
        ``_CHROMA_UPSERT_BATCH_SIZE``-sized slices and each is upserted
        separately. Slicing is by index, so empty slices never occur for a
        non-empty ``ids``.

        Args:
            ids: Chunk IDs to upsert.
            embeddings: Matching embedding vectors.
            documents: Matching chunk contents.
            metadatas: Matching ChromaDB-safe metadata dicts.

        Raises:
            RuntimeError: If any ChromaDB upsert fails.
        """
        if not ids:
            return

        for offset in range(0, len(ids), _CHROMA_UPSERT_BATCH_SIZE):
            slice_end = offset + _CHROMA_UPSERT_BATCH_SIZE
            logger.info(
                f"Upserting chunk batch [{offset}:{slice_end}] "
                f"of {len(ids)}"
            )
            try:
                self._collection.upsert(
                    ids=ids[offset:slice_end],
                    embeddings=embeddings[offset:slice_end],
                    documents=documents[offset:slice_end],
                    metadatas=metadatas[offset:slice_end],
                )
            except Exception as e:
                logger.error(
                    f"Failed to batch upsert chunks "
                    f"[{offset}:{slice_end}]: {e}"
                )
                raise RuntimeError(
                    f"Failed to batch upsert chunks: {e}"
                ) from e

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
