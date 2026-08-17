"""Sprint 6 Vector Store Verification Script.

This script verifies the ChromaDB-backed vector storage layer:
- ChromaDB initialization with configurable persistence
- Single and batch insertion of chunk/embedding pairs
- Count, retrieve-by-ID, and delete operations
- Upsert duplicate handling
- Persistence across store re-initialization
- Metadata preservation (repository_name, etc.)

The output uses ASCII-only status indicators for Windows cp1252
console compatibility.
"""

import hashlib
import logging
import sys
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.chunking.models import ChunkType, CodeChunk
from app.embeddings import EmbeddingEngine, initialize_providers
from app.vectorstore import (
    ChromaVectorStore,
    ChunkEmbeddingPair,
    VectorStoreRecord,
)

# Configure logging (show only warnings and above to keep output clean)
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PASS = "[PASS]"
FAIL = "[FAIL]"


def make_chunk(
    chunk_id: str,
    repository_name: str,
    relative_path: str,
    language: str,
    content: str,
    chunk_type: ChunkType = ChunkType.FUNCTION,
    chunk_index: int = 0,
    start_line: int = 1,
    end_line: int = 10
) -> CodeChunk:
    """Create a CodeChunk with a content hash derived from its content.

    Args:
        chunk_id: Unique chunk identifier
        repository_name: Name of the repository the chunk belongs to
        relative_path: Path relative to the repository root
        language: Programming language of the chunk
        content: The code content
        chunk_type: Type of chunk (function, class, etc.)
        chunk_index: Index within the file
        start_line: Starting line number
        end_line: Ending line number

    Returns:
        A CodeChunk with a valid content hash
    """
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    return CodeChunk(
        id=chunk_id,
        repository_name=repository_name,
        relative_path=relative_path,
        language=language,
        chunk_type=chunk_type,
        chunk_index=chunk_index,
        start_line=start_line,
        end_line=end_line,
        content=content,
        content_hash=content_hash,
        metadata={}
    )


def print_summary(results):
    """Print the final verification summary in ASCII only.

    Args:
        results: List of (description, passed) tuples
    """
    print()
    print("=" * 60)
    print("Sprint 6 Vector Store Verification")
    print("=" * 60)

    all_pass = True
    for description, passed in results:
        status = PASS if passed else FAIL
        print(f"{description}: {status}")
        all_pass = all_pass and passed

    print()
    if all_pass:
        print("Overall Result: ALL TESTS PASSED")
    else:
        print("Overall Result: SOME TESTS FAILED")
    print("=" * 60)
    return all_pass


def verify_vector_store():
    """Run verification tests for the vector store layer.

    Returns:
        True if all tests passed, False otherwise
    """
    results = []

    print()
    print("=" * 60)
    print("Sprint 6 Vector Store Verification")
    print("=" * 60)

    # Use a temporary directory so the demo does not leave
    # persistent database files in the project tree.
    temp_dir = tempfile.mkdtemp(prefix="sprint6_")
    print(f"\nUsing persistence path: {temp_dir}")
    print()

    try:
        # ------------------------------------------------------------
        # Step 1: Initialize embedding system
        # ------------------------------------------------------------
        logger.info("Initializing embedding providers...")
        initialize_providers()
        engine = EmbeddingEngine()

        # ------------------------------------------------------------
        # Step 2: Create realistic code chunks
        # ------------------------------------------------------------
        chunks = [
            make_chunk(
                chunk_id="chunk-auth-0001",
                repository_name="demo-repo",
                relative_path="src/auth.py",
                language="Python",
                content=(
                    "def authenticate_user(username: str, password: str) -> bool:\n"
                    "    \"\"\"Authenticate a user.\"\"\"\n"
                    "    if not username or not password:\n"
                    "        return False\n"
                    "    return password_ok(username, password)\n"
                ),
                chunk_type=ChunkType.FUNCTION,
                chunk_index=0,
                start_line=1,
                end_line=8
            ),
            make_chunk(
                chunk_id="chunk-api-0002",
                repository_name="demo-repo",
                relative_path="src/api/routes.py",
                language="Python",
                content=(
                    "class ApiRouter:\n"
                    "    \"\"\"Routes API requests.\"\"\"\n"
                    "    def __init__(self, handler):\n"
                    "        self._handler = handler\n"
                    "\n"
                    "    def route(self, path: str, method: str):\n"
                    "        return self._handler(path, method)\n"
                ),
                chunk_type=ChunkType.CLASS,
                chunk_index=0,
                start_line=1,
                end_line=9
            ),
            make_chunk(
                chunk_id="chunk-db-0003",
                repository_name="demo-repo",
                relative_path="src/db.py",
                language="Python",
                content=(
                    "def connect(host: str, port: int) -> object:\n"
                    "    \"\"\"Create a database connection.\"\"\"\n"
                    "    return DatabaseClient(host, port)\n"
                ),
                chunk_type=ChunkType.FUNCTION,
                chunk_index=1,
                start_line=1,
                end_line=5
            ),
        ]

        # ------------------------------------------------------------
        # Step 3: Generate embeddings using the existing engine
        # ------------------------------------------------------------
        records = []
        for chunk in chunks:
            record = engine.embed(chunk)
            records.append(record)

        embedding_dim_ok = all(r.dimension == 384 for r in records)
        results.append(("Embedding generation", embedding_dim_ok))
        print(f"Embedding generation: {PASS if embedding_dim_ok else FAIL}")
        if embedding_dim_ok:
            print(f"  {len(records)} embeddings generated, dimension: {records[0].dimension}")

        # ------------------------------------------------------------
        # Step 4: Create the vector store
        # ------------------------------------------------------------
        store = ChromaVectorStore(persistence_path=temp_dir)
        results.append(("ChromaDB initialization", True))
        print(f"ChromaDB initialization: {PASS}")

        # ------------------------------------------------------------
        # Step 5: Store a single chunk/embedding pair
        # ------------------------------------------------------------
        store.add(chunks[0], records[0])
        count_after_single = store.count()
        single_insert_ok = count_after_single == 1
        results.append(("Vector insertion", single_insert_ok))
        print(f"Vector insertion: {PASS if single_insert_ok else FAIL}")
        if not single_insert_ok:
            print(f"  Expected 1 record, got {count_after_single}")

        # ------------------------------------------------------------
        # Step 6: Batch insert remaining chunks
        # ------------------------------------------------------------
        pairs = [
            ChunkEmbeddingPair(chunk=chunks[1], record=records[1]),
            ChunkEmbeddingPair(chunk=chunks[2], record=records[2]),
        ]
        added = store.add_many(pairs)
        count_after_batch = store.count()
        batch_ok = (added == 2) and (count_after_batch == 3)
        results.append(("Batch insertion", batch_ok))
        print(f"Batch insertion: {PASS if batch_ok else FAIL}")
        if not batch_ok:
            print(f"  Expected 2 added, total 3; got added={added}, total={count_after_batch}")

        # ------------------------------------------------------------
        # Step 7: Verify count()
        # ------------------------------------------------------------
        count_ok = store.count() == 3
        results.append(("Count verification", count_ok))
        print(f"Count verification: {PASS if count_ok else FAIL}")
        if not count_ok:
            print(f"  Expected 3, got {store.count()}")

        # ------------------------------------------------------------
        # Step 8: Retrieve by chunk ID and verify fields
        # ------------------------------------------------------------
        retrieved = store.get("chunk-auth-0001")
        get_ok = retrieved is not None
        results.append(("Retrieve by ID", get_ok))
        print(f"Retrieve by ID: {PASS if get_ok else FAIL}")

        if retrieved is not None:
            id_ok = retrieved.id == "chunk-auth-0001"
            doc_ok = retrieved.document == chunks[0].content
            meta_ok = (
                retrieved.metadata.get("repository_name") == "demo-repo"
                and retrieved.metadata.get("relative_path") == "src/auth.py"
                and retrieved.metadata.get("language") == "Python"
                and retrieved.metadata.get("chunk_type") == "function"
                and retrieved.metadata.get("chunk_index") == 0
                and retrieved.metadata.get("start_line") == 1
                and retrieved.metadata.get("end_line") == 8
                and retrieved.metadata.get("content_hash") == chunks[0].content_hash
                and retrieved.metadata.get("model_name") == "BAAI/bge-small-en-v1.5"
            )
            dim_ok = retrieved.embedding is not None and len(retrieved.embedding) == 384

            print(f"  ID match: {PASS if id_ok else FAIL}")
            print(f"  Document match: {PASS if doc_ok else FAIL}")
            print(f"  Metadata match: {PASS if meta_ok else FAIL}")
            print(f"  Embedding dimension (384): {PASS if dim_ok else FAIL}")

            metadata_ok = id_ok and doc_ok and meta_ok and dim_ok
        else:
            metadata_ok = False

        results.append(("Metadata verification", metadata_ok))
        print(f"Metadata verification: {PASS if metadata_ok else FAIL}")

        # ------------------------------------------------------------
        # Step 9: Duplicate/upsert handling
        # ------------------------------------------------------------
        store.add(chunks[0], records[0])  # Insert the same chunk again
        count_after_upsert = store.count()
        upsert_ok = count_after_upsert == 3  # Should NOT increase to 4
        results.append(("Duplicate/upsert verification", upsert_ok))
        print(f"Duplicate/upsert verification: {PASS if upsert_ok else FAIL}")
        if not upsert_ok:
            print(f"  Expected count 3 after upsert, got {count_after_upsert}")

        # ------------------------------------------------------------
        # Step 10: Persistence across store re-initialization
        # ------------------------------------------------------------
        reopened_store = ChromaVectorStore(persistence_path=temp_dir)
        reopened_count = reopened_store.count()
        persistence_ok = reopened_count == 3

        if persistence_ok:
            reopened_record = reopened_store.get("chunk-api-0002")
            persistence_ok = (
                reopened_record is not None
                and reopened_record.document == chunks[1].content
            )

        results.append(("Persistence verification", persistence_ok))
        print(f"Persistence verification: {PASS if persistence_ok else FAIL}")
        if not persistence_ok:
            print(f"  Expected 3 records after reopen, got {reopened_count}")

        # ------------------------------------------------------------
        # Step 11: Delete one chunk and verify count decreases
        # ------------------------------------------------------------
        deleted = store.delete("chunk-db-0003")
        count_after_delete = store.count()
        delete_ok = deleted and count_after_delete == 2
        results.append(("Delete verification", delete_ok))
        print(f"Delete verification: {PASS if delete_ok else FAIL}")
        if not delete_ok:
            print(f"  deleted={deleted}, expected count 2, got {count_after_delete}")

        # Verify deleted record is actually gone
        missing = store.get("chunk-db-0003")
        if missing is None:
            print(f"  Deleted record confirmed absent: {PASS}")
        else:
            print(f"  Deleted record confirmed absent: {FAIL}")

        # ------------------------------------------------------------
        # Cleanup: remove the temporary database
        # ------------------------------------------------------------
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        logger.error(f"Verification failed with error: {e}", exc_info=True)
        print(f"\n[ERROR] Verification failed - {e}\n")
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        return False

    # ------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------
    return print_summary(results)


if __name__ == "__main__":
    success = verify_vector_store()
    sys.exit(0 if success else 1)
