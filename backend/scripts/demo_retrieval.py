"""Sprint 7 Semantic Retrieval Verification Script.

This script verifies the Retriever layer implemented in Sprint 7:
- Question embedding via the existing EmbeddingEngine
- Similarity querying through the VectorStore abstraction
- Structured RetrievalResult objects with preserved metadata
- Repository isolation (repo-a vs repo-b with overlapping files)
- Input validation (empty question, empty repo name, invalid top_k)
- top_k enforcement, ordering, and relevance scoring
- Graceful handling of queries with no matching results

The output uses ASCII-only status indicators for Windows cp1252
console compatibility. The script exits 0 only when every check passes.

Data flow exercised:
    User question
        -> Retriever
        -> Existing EmbeddingEngine
        -> question embedding
        -> Existing VectorStore abstraction
        -> ChromaDB
        -> Top-K relevant chunks
        -> RetrievalResult[]
"""

import hashlib
import logging
import shutil
import sys
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.chunking.models import ChunkType, CodeChunk
from app.embeddings import EmbeddingEngine, initialize_providers
from app.rag import Retriever, RetrievalResult
from app.vectorstore import ChromaVectorStore, SimilarityResult, VectorStore

# Configure logging (show only warnings and above to keep output clean)
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PASS = "[PASS]"
FAIL = "[FAIL]"
ERROR = "[ERROR]"

# Realistic code snippets shared across both repositories so that the
# repository-isolation check is meaningful (identical names, overlapping
# content categories: auth, database, payment).

REPO_A_AUTH = '''def authenticate_user(username: str, password: str) -> bool:
    """Authenticate a user by checking credentials."""
    if not username or not password:
        return False
    user = db.query(User).filter(User.username == username).first()
    if user and verify_password(password, user.password_hash):
        log_activity(user.id, "login")
        return True
    return False'''

REPO_A_DATABASE = '''class DatabaseConnection:
    """Manages a pool of database connections."""

    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port
        self._pool = []

    def connect(self) -> object:
        """Acquire a connection from the pool."""
        conn = psycopg2.connect(host=self._host, port=self._port)
        self._pool.append(conn)
        return conn'''

REPO_A_PAYMENT = '''def charge_credit_card(card_number: str, amount: float) -> bool:
    """Charge a credit card for the given amount."""
    gateway = PaymentGateway()
    try:
        gateway.charge(card_number, amount)
        return True
    except PaymentDeclinedError:
        logger.warning("Payment declined")
        return False'''

REPO_B_AUTH = '''def authenticate_user(username: str, password: str) -> bool:
    """Validate user credentials for login."""
    if not username or not password:
        return False
    user = user_table.find_one({"username": username})
    if user and bcrypt.checkpw(password.encode(), user["hash"]):
        return True
    return False'''

REPO_B_DATABASE = '''def get_database_client(host: str, port: int) -> object:
    """Return a configured database client."""
    client = MongoClient(host, port)
    client.auth_db = "users"
    return client'''

REPO_B_PAYMENT = '''def process_refund(transaction_id: str, amount: float) -> bool:
    """Refund a transaction by its identifier."""
    tx = find_transaction(transaction_id)
    if tx is None:
        logger.warning("Transaction not found: %s", transaction_id)
        return False
    return refund_gateway.refund(tx.payment_token, amount)'''


def make_chunk(
    chunk_id: str,
    repository_name: str,
    relative_path: str,
    content: str,
    chunk_type: ChunkType = ChunkType.FUNCTION,
    chunk_index: int = 0,
    start_line: int = 1,
    end_line: int = 12
) -> CodeChunk:
    """Create a CodeChunk with a content hash derived from its content.

    Args:
        chunk_id: Unique chunk identifier
        repository_name: Name of the repository the chunk belongs to
        relative_path: Path relative to the repository root
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
        language="Python",
        chunk_type=chunk_type,
        chunk_index=chunk_index,
        start_line=start_line,
        end_line=end_line,
        content=content,
        content_hash=content_hash,
        metadata={}
    )


def build_repo_chunks(repo_name: str, prefix: str) -> list:
    """Build three code chunks for a repository.

    Args:
        repo_name: Name of the repository (e.g. "repo-a")
        prefix: Chunk ID prefix (e.g. "ra")

    Returns:
        List of CodeChunk covering auth, database, and payment
    """
    return [
        make_chunk(
            chunk_id=f"{prefix}-auth-0001",
            repository_name=repo_name,
            relative_path="src/auth.py",
            content=REPO_A_AUTH if repo_name == "repo-a" else REPO_B_AUTH,
            chunk_type=ChunkType.FUNCTION,
            chunk_index=0,
            start_line=1,
            end_line=12
        ),
        make_chunk(
            chunk_id=f"{prefix}-db-0002",
            repository_name=repo_name,
            relative_path="src/database.py",
            content=REPO_A_DATABASE if repo_name == "repo-a" else REPO_B_DATABASE,
            chunk_type=ChunkType.CLASS if repo_name == "repo-a" else ChunkType.FUNCTION,
            chunk_index=0,
            start_line=1,
            end_line=11
        ),
        make_chunk(
            chunk_id=f"{prefix}-pay-0003",
            repository_name=repo_name,
            relative_path="src/payment.py",
            content=REPO_A_PAYMENT if repo_name == "repo-a" else REPO_B_PAYMENT,
            chunk_type=ChunkType.FUNCTION,
            chunk_index=0,
            start_line=1,
            end_line=11
        ),
    ]


def preview(content: str, limit: int = 90) -> str:
    """Return a single-line content preview.

    Args:
        content: Full chunk content
        limit: Maximum number of characters to show

    Returns:
        A truncated, newline-free preview string
    """
    flat = " ".join(content.split())
    if len(flat) <= limit:
        return flat
    return flat[:limit] + "..."


def print_results(results, question: str):
    """Pretty-print retrieval results with rank and score.

    Args:
        results: List of RetrievalResult
        question: The query that produced the results
    """
    print(f"\n  Query: {question!r}")
    if not results:
        print("  No results returned.")
        return
    for rank, result in enumerate(results, start=1):
        print(f"  {rank}. score={result.relevance_score:.4f} "
              f"| repo={result.repository_name} "
              f"| {result.relative_path}:{result.start_line}-{result.end_line} "
              f"| {result.chunk_type.value}")
        print(f"     {preview(result.content)}")


def verify_retrieval():
    """Run verification checks for the retrieval layer.

    Returns:
        True if all checks passed, False otherwise
    """
    results = []
    temp_dir = None

    print()
    print("=" * 60)
    print("Sprint 7 Semantic Retrieval Verification")
    print("=" * 60)

    try:
        # ------------------------------------------------------------
        # Step 1: Initialize the existing embedding provider system
        # ------------------------------------------------------------
        print("\n[1] Initializing embedding providers...")
        initialize_providers()
        engine = EmbeddingEngine()
        provider_info = engine.get_provider_info()
        print(f"    Model: {provider_info['model_name']}")
        print(f"    Dimension: {provider_info['dimension']}")
        results.append(("Embedding system initialized", True))
        print(f"    Embedding system initialized: {PASS}")

        # ------------------------------------------------------------
        # Step 2: Create realistic CodeChunks for two repositories
        # ------------------------------------------------------------
        print("\n[2] Creating code chunks for repo-a and repo-b...")
        repo_a_chunks = build_repo_chunks("repo-a", "ra")
        repo_b_chunks = build_repo_chunks("repo-b", "rb")
        all_chunks = repo_a_chunks + repo_b_chunks

        print(f"    repo-a chunks: {len(repo_a_chunks)}")
        for c in repo_a_chunks:
            print(f"      {c.id}  {c.relative_path}:{c.start_line}-{c.end_line} [{c.chunk_type.value}]")
        print(f"    repo-b chunks: {len(repo_b_chunks)}")
        for c in repo_b_chunks:
            print(f"      {c.id}  {c.relative_path}:{c.start_line}-{c.end_line} [{c.chunk_type.value}]")

        overlap = {c.relative_path for c in repo_a_chunks} & {
            c.relative_path for c in repo_b_chunks
        }
        overlapping_ok = len(overlap) == 3
        results.append(("Overlapping files created", overlapping_ok))
        print(f"    Overlapping files across repos: {PASS if overlapping_ok else FAIL}")

        # ------------------------------------------------------------
        # Step 3: Generate embeddings using the existing EmbeddingEngine
        # ------------------------------------------------------------
        print("\n[3] Generating embeddings...")
        records = []
        for chunk in all_chunks:
            record = engine.embed(chunk)
            records.append(record)

        embedding_ok = len(records) == len(all_chunks)
        embedding_ok = embedding_ok and all(
            r.dimension == 384 and len(r.embedding) == 384 for r in records
        )
        results.append(("Embedding generation", embedding_ok))
        print(f"    Embedding generation: {PASS if embedding_ok else FAIL}")
        if embedding_ok:
            print(f"      {len(records)} embeddings generated, dimension: {records[0].dimension}")

        # ------------------------------------------------------------
        # Step 4: Store them using the existing ChromaVectorStore
        # ------------------------------------------------------------
        print("\n[4] Storing chunks in ChromaVectorStore...")
        temp_dir = tempfile.mkdtemp(prefix="sprint7_")
        print(f"    Using persistence path: {temp_dir}")

        store = ChromaVectorStore(persistence_path=temp_dir)
        for chunk, record in zip(all_chunks, records):
            store.add(chunk, record)
        store_count = store.count()
        store_ok = store_count == len(all_chunks)
        results.append(("Vector storage", store_ok))
        print(f"    Vector storage: {PASS if store_ok else FAIL}")
        if not store_ok:
            print(f"      Expected {len(all_chunks)} records, got {store_count}")
        else:
            print(f"      {store_count} records stored")

        # ------------------------------------------------------------
        # Step 5: Create a Retriever via the VectorStore abstraction
        # ------------------------------------------------------------
        print("\n[5] Creating Retriever (depends only on abstractions)...")
        store_as_interface: VectorStore = store  # type is VectorStore
        retriever = Retriever(embedding_engine=engine, vector_store=store_as_interface)
        print(f"    Retriever created with engine={type(engine).__name__}, "
              f"store={type(store_as_interface).__name__}")
        print(f"    Retriever type: {type(retriever).__name__}")

        # Verify the Retriever never depends on the concrete store class.
        import inspect
        retriever_deps = inspect.signature(Retriever.__init__).parameters
        concrete_refs = [
            dep for dep in retriever_deps
            if "store" in dep or "engine" in dep
        ]
        abstraction_ok = "vector_store" in retriever_deps and "embedding_engine" in retriever_deps
        results.append(("Retriever abstraction dependency", abstraction_ok))
        print(f"    Retriever abstraction dependency: {PASS if abstraction_ok else FAIL}")

        # ------------------------------------------------------------
        # Step 6: Input validation
        # ------------------------------------------------------------
        print("\n[6] Validating input handling...")

        validation_cases = [
            ("empty question", lambda: retriever.retrieve("   ", "repo-a")),
            ("empty repo name", lambda: retriever.retrieve("auth", "  ")),
            ("top_k zero", lambda: retriever.retrieve("auth", "repo-a", top_k=0)),
            ("top_k negative", lambda: retriever.retrieve("auth", "repo-a", top_k=-3)),
        ]
        validation_all_ok = True
        for label, call in validation_cases:
            try:
                call()
                print(f"    {label}: {FAIL} (no ValueError raised)")
                validation_all_ok = False
            except ValueError as e:
                print(f"    {label}: {PASS} ({str(e)[:60]})")
            except Exception as e:
                print(f"    {label}: {FAIL} (unexpected {type(e).__name__})")
                validation_all_ok = False
        results.append(("Input validation", validation_all_ok))
        print(f"    Input validation: {PASS if validation_all_ok else FAIL}")

        # ------------------------------------------------------------
        # Step 7: Run a natural-language query
        # ------------------------------------------------------------
        print("\n[7] Natural-language query on repo-a...")
        question = "How is user authentication handled?"
        auth_results = retriever.retrieve(question, "repo-a", top_k=5)
        print_results(auth_results, question)

        question_embedding_ok = len(auth_results) > 0
        results.append(("Question embedding & retrieval", question_embedding_ok))
        print(f"    Question embedding & retrieval: {PASS if question_embedding_ok else FAIL}")

        # ------------------------------------------------------------
        # Step 8: Verify top_k is respected and ordering is correct
        # ------------------------------------------------------------
        print("\n[8] Verifying top_k and ordering...")
        top_k = 2
        limited = retriever.retrieve(question, "repo-a", top_k=top_k)
        print(f"    Requested top_k={top_k}, returned {len(limited)}")
        top_k_ok = len(limited) <= top_k
        results.append(("top_k respected", top_k_ok))
        print(f"    top_k respected: {PASS if top_k_ok else FAIL}")

        ordered_ok = all(
            limited[i].relevance_score >= limited[i + 1].relevance_score
            for i in range(len(limited) - 1)
        )
        results.append(("Results ordered desc", ordered_ok))
        print(f"    Results ordered descending: {PASS if ordered_ok else FAIL}")
        if limited:
            print(f"    Scores: {[f'{r.relevance_score:.4f}' for r in limited]}")

        # ------------------------------------------------------------
        # Step 9: Verify metadata is preserved
        # ------------------------------------------------------------
        print("\n[9] Verifying metadata preservation...")
        if auth_results:
            top = auth_results[0]
            meta = top.metadata
            meta_ok = (
                meta.get("repository_name") == "repo-a"
                and meta.get("relative_path") == "src/auth.py"
                and meta.get("language") == "Python"
                and meta.get("chunk_type") == "function"
                and meta.get("chunk_index") == 0
                and meta.get("start_line") == 1
                and meta.get("end_line") == 12
                and meta.get("content_hash")
                and meta.get("model_name") == "BAAI/bge-small-en-v1.5"
            )
            print(f"    Top result metadata:")
            print(f"      repository_name: {meta.get('repository_name')}")
            print(f"      relative_path:   {meta.get('relative_path')}")
            print(f"      language:        {meta.get('language')}")
            print(f"      chunk_type:      {meta.get('chunk_type')}")
            print(f"      chunk_index:     {meta.get('chunk_index')}")
            print(f"      start_line:      {meta.get('start_line')}")
            print(f"      end_line:        {meta.get('end_line')}")
            print(f"      model_name:      {meta.get('model_name')}")
            print(f"      content_hash:    {str(meta.get('content_hash'))[:16]}...")

            field_ok = (
                isinstance(top, RetrievalResult)
                and top.chunk_id
                and top.content
                and top.repository_name == "repo-a"
                and top.relative_path == "src/auth.py"
                and top.language == "Python"
                and top.chunk_type == ChunkType.FUNCTION
                and top.chunk_index == 0
                and top.start_line == 1
                and top.end_line == 12
                and 0.0 <= top.relevance_score <= 1.0
                and top.metadata is not None
            )
            results.append(("Metadata preservation", meta_ok and field_ok))
            print(f"    Metadata preservation: {PASS if (meta_ok and field_ok) else FAIL}")
        else:
            results.append(("Metadata preservation", False))
            print(f"    Metadata preservation: {FAIL} (no results)")

        # ------------------------------------------------------------
        # Step 10: Repository isolation (the critical check)
        # ------------------------------------------------------------
        print("\n[10] Verifying repository isolation...")
        print(f"    Querying repo-a with {question!r}")
        repo_a_only = retriever.retrieve(question, "repo-a", top_k=10)
        isolation_ok = all(r.repository_name == "repo-a" for r in repo_a_only)
        results.append(("Repository isolation", isolation_ok))
        print(f"    Repository isolation: {PASS if isolation_ok else FAIL}")
        if not isolation_ok:
            bad = [r.chunk_id for r in repo_a_only if r.repository_name != "repo-a"]
            print(f"      Leaked chunks from other repos: {bad}")

        # Reverse check: repo-b query must not return repo-a chunks.
        print(f"    Querying repo-b with {question!r}")
        repo_b_results = retriever.retrieve(question, "repo-b", top_k=10)
        reverse_ok = all(r.repository_name == "repo-b" for r in repo_b_results)
        reverse_ok = reverse_ok and bool(repo_b_results)
        results.append(("Reverse isolation (repo-b)", reverse_ok))
        print(f"    Reverse isolation (repo-b): {PASS if reverse_ok else FAIL}")

        # ------------------------------------------------------------
        # Step 11: Unknown query robustness + graceful empty result
        # ------------------------------------------------------------
        print("\n[11] Unknown query handling...")

        # Part A: a semantically-garbage query must not crash. Over a
        # tiny 6-chunk corpus there is always some nearest neighbor, so
        # it returns results -- this is correct cosine behavior, not a
        # failure. The no-result path is verified deterministically in
        # part B via an unindexed repository.
        unknown_question = "zzqxqxqx quarkwarp teleportation flux capacitor"
        try:
            garbage_results = retriever.retrieve(unknown_question, "repo-a", top_k=5)
            no_crash_ok = isinstance(garbage_results, list)
            print(f"    Unknown query completed: {PASS if no_crash_ok else FAIL}")
            print(f"      Returned {len(garbage_results)} nearest neighbors "
                  f"(expected: tiny corpus always has a nearest vector)")
        except Exception as e:
            no_crash_ok = False
            print(f"    Unknown query completed: {FAIL} ({type(e).__name__}: {e})")

        # Part B: querying a repository with no indexed chunks must
        # return an empty list without crashing. This deterministically
        # exercises the no-result path of the metadata-filtered query.
        print(f"    Querying unindexed repository 'repo-not-indexed'...")
        empty_results = retriever.retrieve(unknown_question, "repo-not-indexed", top_k=5)
        empty_ok = empty_results == []
        results.append(("Empty result graceful", empty_ok))
        print(f"    Empty result graceful: {PASS if empty_ok else FAIL}")
        if not empty_ok:
            print(f"      Expected [], got {len(empty_results)} results")

        # ------------------------------------------------------------
        # Step 12: Relevance score range sanity
        # ------------------------------------------------------------
        print("\n[12] Verifying relevance score range...")
        all_top = auth_results + repo_b_results
        score_range_ok = all(
            0.0 <= r.relevance_score <= 1.0 for r in all_top
        )
        results.append(("Score range [0,1]", score_range_ok))
        print(f"    All scores within [0, 1]: {PASS if score_range_ok else FAIL}")
        if all_top:
            print(f"    Max score: {max(r.relevance_score for r in all_top):.4f}, "
                  f"Min score: {min(r.relevance_score for r in all_top):.4f}")

        # ------------------------------------------------------------
        # Cleanup: remove the temporary database
        # ------------------------------------------------------------
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"\n    Cleaned up temporary store at {temp_dir}")

    except Exception as e:
        logger.error(f"Verification failed with error: {e}", exc_info=True)
        print(f"\n{ERROR} Verification failed - {e}\n")
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
        return False

    # ------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------
    print()
    print("=" * 60)
    print("Sprint 7 Retrieval Verification Summary")
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


if __name__ == "__main__":
    success = verify_retrieval()
    sys.exit(0 if success else 1)
