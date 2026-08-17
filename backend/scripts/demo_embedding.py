"""Sprint 5 Embedding Engine Verification Script.

This script demonstrates and verifies the core functionality of the
EmbeddingEngine implemented in Sprint 5, including:
- Embedding generation for code chunks
- Cache functionality
- Vector dimension validation
"""

import hashlib
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.chunking.models import ChunkType, CodeChunk
from app.embeddings import initialize_providers
from app.embeddings.engine import EmbeddingEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_demo_chunk() -> CodeChunk:
    """Create a realistic CodeChunk for demonstration.

    Returns:
        A CodeChunk representing a Python authentication function
    """
    content = '''def authenticate_user(username: str, password: str) -> bool:
    """Authenticate a user with username and password.

    Args:
        username: The username to authenticate
        password: The password to verify

    Returns:
        True if authentication successful, False otherwise
    """
    if not username or not password:
        return False

    # Hash password for comparison
    password_hash = hashlib.sha256(password.encode()).hexdigest()

    # Query database for user
    user = db.query(User).filter(User.username == username).first()

    if user and user.password_hash == password_hash:
        logger.info(f"User {username} authenticated successfully")
        return True

    logger.warning(f"Authentication failed for user {username}")
    return False'''

    # Generate content hash
    content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()

    chunk = CodeChunk(
        id="demo-chunk-550e8400-e29b-41d4-a716-446655440000",
        repository_name="demo-repository",
        relative_path="src/auth.py",
        language="Python",
        chunk_type=ChunkType.FUNCTION,
        chunk_index=0,
        start_line=10,
        end_line=35,
        content=content,
        token_count=150,
        content_hash=content_hash,
        metadata={
            "function_name": "authenticate_user",
            "has_docstring": True,
            "complexity": "medium"
        }
    )

    return chunk


def verify_embedding_engine():
    """Run verification tests for the EmbeddingEngine."""

    print("\n" + "=" * 60)
    print("Sprint 5 Embedding Engine Verification")
    print("=" * 60 + "\n")

    try:
        # Step 1: Create a demo code chunk
        logger.info("Creating demo code chunk...")
        chunk = create_demo_chunk()

        print(f"Created CodeChunk:")
        print(f"  ID: {chunk.id}")
        print(f"  Repository: {chunk.repository_name}")
        print(f"  File: {chunk.relative_path}")
        print(f"  Language: {chunk.language}")
        print(f"  Type: {chunk.chunk_type}")
        print(f"  Lines: {chunk.start_line}-{chunk.end_line}")
        print(f"  Content Hash: {chunk.content_hash[:16]}...")
        print(f"  Function: {chunk.metadata.get('function_name')}")
        print()

        # Step 2: Initialize the EmbeddingEngine
        logger.info("Initializing provider registry...")
        initialize_providers()

        logger.info("Initializing EmbeddingEngine...")
        engine = EmbeddingEngine()

        # Get provider info
        provider_info = engine.get_provider_info()
        print(f"Provider Information:")
        print(f"  Model: {provider_info['model_name']}")
        print(f"  Dimension: {provider_info['dimension']}")
        print(f"  Provider Class: {provider_info['provider_class']}")
        print()

        # Step 3: Capture cache state before first embedding
        logger.info("Capturing cache state before first embedding...")
        cache_stats_before = engine.get_cache_stats()
        print(f"Cache Statistics (before first call):")
        print(f"  Total Entries: {cache_stats_before['total_entries']}")
        print(f"  Total Hashes: {cache_stats_before['total_hashes']}")
        print()

        # Step 4: Generate first embedding
        logger.info("Generating first embedding...")
        print("Generating embedding (first call)...")
        record1 = engine.embed(chunk)

        # Capture cache state after first embedding
        cache_stats_after_first = engine.get_cache_stats()

        print(f"\nEmbedding Record (First Call):")
        print(f"  Record ID: {record1.id}")
        print(f"  Chunk ID: {record1.chunk_id}")
        print(f"  Model Name: {record1.model_name}")
        print(f"  Dimension: {record1.dimension}")
        print(f"  Actual Vector Length: {len(record1.embedding)}")
        print(f"  Content Hash: {record1.content_hash[:16]}...")
        print(f"  Has Embedding: {record1.embedding is not None and len(record1.embedding) > 0}")
        print(f"  First 10 values: {record1.embedding[:10]}")
        print()

        # Step 5: Verify dimension matches vector length
        dimension_match = len(record1.embedding) == record1.dimension
        print(f"\nDimension Verification:")
        print(f"  Expected Dimension: {record1.dimension}")
        print(f"  Actual Vector Length: {len(record1.embedding)}")
        print(f"  Match: {'[PASS]' if dimension_match else '[FAIL]'}")
        print()

        # Step 6: Verify dimension is 384 for BGE model
        expected_bge_dimension = 384
        bge_dimension_match = record1.dimension == expected_bge_dimension
        print(f"BGE Model Dimension Verification:")
        print(f"  Expected for BGE: {expected_bge_dimension}")
        print(f"  Actual: {record1.dimension}")
        print(f"  Match: {'[PASS]' if bge_dimension_match else '[FAIL]'}")
        print()

        # Step 7: Test caching - generate embedding again with same chunk
        logger.info("Testing cache functionality...")
        print("Cache Statistics (after first call):")
        print(f"  Total Entries: {cache_stats_after_first['total_entries']}")
        print(f"  Total Hashes: {cache_stats_after_first['total_hashes']}")
        print()

        print("Generating embedding (second call - should use cache)...")
        record2 = engine.embed(chunk)

        # Capture cache state after second embedding
        cache_stats_after_second = engine.get_cache_stats()
        print(f"\nCache Statistics (after second call):")
        print(f"  Total Entries: {cache_stats_after_second['total_entries']}")
        print(f"  Total Hashes: {cache_stats_after_second['total_hashes']}")
        print()

        # Step 8: Verify cache was used (entries should NOT increase)
        entries_before = cache_stats_after_first['total_entries']
        entries_after = cache_stats_after_second['total_entries']
        hashes_before = cache_stats_after_first['total_hashes']
        hashes_after = cache_stats_after_second['total_hashes']

        cache_reused = (entries_after == entries_before) and (hashes_after == hashes_before)
        vectors_identical = record1.embedding == record2.embedding
        content_hash_match = record1.content_hash == record2.content_hash
        same_object = record1 is record2

        print(f"Cache Verification:")
        print(f"  Entries before first call: {cache_stats_before['total_entries']}")
        print(f"  Entries after first call: {entries_before}")
        print(f"  Entries after second call: {entries_after}")
        print(f"  Cache reused (entries unchanged): {'[PASS]' if cache_reused else '[FAIL]'}")
        print(f"  Vectors identical: {'[PASS]' if vectors_identical else '[FAIL]'}")
        print(f"  Content hash match: {'[PASS]' if content_hash_match else '[FAIL]'}")
        print(f"  Same cached record object: {'[PASS]' if same_object else '[FAIL]'}")

        if not vectors_identical:
            # Calculate how many values differ
            differences = sum(1 for a, b in zip(record1.embedding, record2.embedding) if a != b)
            print(f"  Number of differing values: {differences}/{len(record1.embedding)}")
        print()

        # Final Summary
        print("=" * 60)
        print("Sprint 5 Embedding Verification Summary")
        print("=" * 60)
        print(f"Model: {record1.model_name}")
        print(f"Dimension: {record1.dimension}")
        print(f"Vector Length: {len(record1.embedding)}")
        print(f"Dimension Match: {'[PASS]' if dimension_match else '[FAIL]'}")
        print(f"BGE Dimension (384): {'[PASS]' if bge_dimension_match else '[FAIL]'}")
        print(f"Cache Reused: {'[PASS]' if cache_reused else '[FAIL]'}")
        print(f"Vectors Identical: {'[PASS]' if vectors_identical else '[FAIL]'}")
        print(f"Content Hash Match: {'[PASS]' if content_hash_match else '[FAIL]'}")
        print(f"Same Object: {'[PASS]' if same_object else '[FAIL]'}")

        # Overall pass/fail
        all_pass = (dimension_match and bge_dimension_match and cache_reused and
                    vectors_identical and content_hash_match and same_object)
        print(f"\nOverall Result: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
        print("=" * 60 + "\n")

        return all_pass

    except Exception as e:
        logger.error(f"Verification failed with error: {e}", exc_info=True)
        print(f"\n[ERROR] Verification failed - {e}\n")
        return False


if __name__ == "__main__":
    success = verify_embedding_engine()
    sys.exit(0 if success else 1)
