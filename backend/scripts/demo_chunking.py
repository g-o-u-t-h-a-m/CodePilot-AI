"""
CodePilot AI - Chunk Engine Demonstration Script

This script demonstrates the chunk engine capabilities by:
1. Scanning a repository using RepositoryScanner
2. Processing files through ChunkEngine
3. Displaying comprehensive statistics and samples
"""

import sys
import time
import logging
from pathlib import Path
from collections import Counter
from typing import List
import io

# Fix Windows console encoding for Unicode support
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.repository.scanner import RepositoryScanner
from app.chunking import ChunkEngine, initialize_strategies
from app.models.source_file import SourceFile
from app.chunking.models import CodeChunk


# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_header():
    """Print the demonstration header."""
    print("=" * 50)
    print("\nCodePilot AI")
    print("Chunk Engine Demonstration\n")
    print("=" * 50)


def print_summary(repo_name: str, files_count: int, chunks_count: int, time_taken: float):
    """Print processing summary statistics."""
    print(f"\nRepository Name: {repo_name}")
    print(f"Files Scanned: {files_count}")
    print(f"Chunks Generated: {chunks_count}")
    print(f"Time Taken: {time_taken:.2f}s")


def print_language_distribution(source_files: List[SourceFile]):
    """Print language distribution statistics."""
    print("\n" + "=" * 50)
    print("Language Distribution")
    print("=" * 50)

    language_counts = Counter(file.language for file in source_files)

    for language, count in sorted(language_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(source_files)) * 100
        print(f"{language:20s}: {count:4d} files ({percentage:5.1f}%)")


def print_chunk_type_distribution(chunks: List[CodeChunk]):
    """Print chunk type distribution statistics."""
    print("\n" + "=" * 50)
    print("Chunk Type Distribution")
    print("=" * 50)

    chunk_type_counts = Counter(chunk.chunk_type.value for chunk in chunks)

    for chunk_type, count in sorted(chunk_type_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(chunks)) * 100
        print(f"{chunk_type:20s}: {count:4d} chunks ({percentage:5.1f}%)")


def print_sample_chunks(chunks: List[CodeChunk], sample_size: int = 5):
    """Print sample chunks with details."""
    print("\n" + "=" * 50)
    print("Sample Chunks")
    print("=" * 50)

    for i, chunk in enumerate(chunks[:sample_size], 1):
        print(f"\n--- Chunk {i} ---")
        print(f"Chunk ID: {chunk.id}")
        print(f"Language: {chunk.language}")
        print(f"Chunk Type: {chunk.chunk_type.value}")
        print(f"Relative Path: {chunk.relative_path}")
        print(f"Start Line: {chunk.start_line}")
        print(f"End Line: {chunk.end_line}")

        # Content preview (first 100 characters)
        content_preview = chunk.content.replace('\n', ' ')[:100]
        if len(chunk.content) > 100:
            content_preview += "..."
        print(f"Content Preview: {content_preview}")


def find_repository_path() -> Path:
    """Find a repository to demonstrate chunking.

    Priority:
    1. Command line argument
    2. First directory in indexed_repos/
    3. Backend directory as fallback

    Returns:
        Path to the repository
    """
    # Check command line argument
    if len(sys.argv) > 1:
        repo_path = Path(sys.argv[1])
        if repo_path.exists() and repo_path.is_dir():
            return repo_path
        else:
            logger.warning(f"Provided path does not exist: {sys.argv[1]}")

    # Check indexed_repos directory
    indexed_repos_dir = Path(__file__).parent.parent.parent / "indexed_repos"
    if indexed_repos_dir.exists():
        subdirs = [d for d in indexed_repos_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        if subdirs:
            return subdirs[0]

    # Fallback to backend directory
    backend_dir = Path(__file__).parent.parent
    logger.info(f"Using backend directory as demonstration: {backend_dir}")
    return backend_dir


def main():
    """Main demonstration execution."""
    try:
        print_header()

        # Find repository to scan
        repo_path = find_repository_path()
        repo_name = repo_path.name

        print(f"\nScanning repository: {repo_path}")
        print("Please wait...\n")

        # Start timing
        start_time = time.time()

        # Initialize chunking strategies
        initialize_strategies()

        # Initialize scanner and engine
        scanner = RepositoryScanner()
        engine = ChunkEngine()

        # Scan repository
        logger.info(f"Scanning repository: {repo_path}")
        source_files = scanner.scan(str(repo_path))

        if not source_files:
            print("\nNo source files found in the repository.")
            print("Make sure the repository contains supported file types.")
            return 1

        # Process files through chunk engine
        logger.info(f"Processing {len(source_files)} files through chunk engine")
        chunks = engine.chunk_files(source_files)

        if not chunks:
            print("\nNo chunks generated from source files.")
            print("This may indicate all files were skipped or unsupported.")
            return 1

        # Calculate time taken
        time_taken = time.time() - start_time

        # Print statistics
        print_summary(repo_name, len(source_files), len(chunks), time_taken)
        print_language_distribution(source_files)
        print_chunk_type_distribution(chunks)
        print_sample_chunks(chunks)

        # Final separator
        print("\n" + "=" * 50)
        print("\nDemonstration completed successfully!")

        return 0

    except KeyboardInterrupt:
        print("\n\nDemonstration interrupted by user.")
        return 130

    except Exception as e:
        logger.error(f"Demonstration failed: {e}", exc_info=True)
        print(f"\nError: {e}")
        print("\nDemonstration failed. Check logs for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
