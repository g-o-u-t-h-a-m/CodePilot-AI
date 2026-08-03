"""Test script for the Chunk Engine.

This script demonstrates the Chunk Engine functionality by chunking
various file types and displaying the results.
"""

import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from app.chunking import ChunkEngine, initialize_strategies
from app.models.source_file import SourceFile
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def create_sample_python_file() -> SourceFile:
    """Create a sample Python source file for testing."""
    content = '''def hello():
    """Say hello."""
    print("Hello, World!")

class Calculator:
    """Simple calculator class."""

    def add(self, a, b):
        """Add two numbers."""
        return a + b

    def subtract(self, a, b):
        """Subtract two numbers."""
        return a - b
'''

    return SourceFile(
        path="/test/indexed_repos/test-repo/calculator.py",
        relative_path="calculator.py",
        extension=".py",
        language="Python",
        size=len(content),
        line_count=len(content.splitlines()),
        encoding="utf-8",
        sha256="test-hash",
        content=content
    )


def create_sample_markdown_file() -> SourceFile:
    """Create a sample Markdown source file for testing."""
    content = '''# Introduction

This is the introduction section.

## Getting Started

Follow these steps to get started.

## Advanced Topics

This section covers advanced topics.

### Sub-topic

Details about the sub-topic.
'''

    return SourceFile(
        path="/test/indexed_repos/test-repo/README.md",
        relative_path="README.md",
        extension=".md",
        language="Markdown",
        size=len(content),
        line_count=len(content.splitlines()),
        encoding="utf-8",
        sha256="test-hash",
        content=content
    )


def create_sample_json_file() -> SourceFile:
    """Create a sample JSON source file for testing."""
    content = '''{
  "name": "test-project",
  "version": "1.0.0",
  "dependencies": {
    "express": "^4.18.0"
  }
}'''

    return SourceFile(
        path="/test/indexed_repos/test-repo/package.json",
        relative_path="package.json",
        extension=".json",
        language="JSON",
        size=len(content),
        line_count=len(content.splitlines()),
        encoding="utf-8",
        sha256="test-hash",
        content=content
    )


def create_sample_javascript_file() -> SourceFile:
    """Create a sample JavaScript source file for testing."""
    content = '''function greet(name) {
    return `Hello, ${name}!`;
}

const add = (a, b) => {
    return a + b;
};

class UserManager {
    constructor() {
        this.users = [];
    }

    addUser(user) {
        this.users.push(user);
    }
}

export { greet, add, UserManager };
'''

    return SourceFile(
        path="/test/indexed_repos/test-repo/utils.js",
        relative_path="utils.js",
        extension=".js",
        language="JavaScript",
        size=len(content),
        line_count=len(content.splitlines()),
        encoding="utf-8",
        sha256="test-hash",
        content=content
    )


def print_chunk_summary(chunks, file_type):
    """Print a summary of chunks."""
    print(f"\n{'=' * 60}")
    print(f"{file_type} File Chunking Results")
    print(f"{'=' * 60}")
    print(f"Total chunks: {len(chunks)}\n")

    for chunk in chunks:
        print(f"Chunk {chunk.chunk_index}:")
        print(f"  Type: {chunk.chunk_type.value}")
        print(f"  Lines: {chunk.start_line}-{chunk.end_line}")
        print(f"  Hash: {chunk.content_hash[:16]}...")
        if chunk.metadata:
            print(f"  Metadata: {chunk.metadata}")
        print()


def main():
    """Run the test."""
    print("Initializing Chunk Engine...")

    # Initialize strategies
    initialize_strategies()

    # Create engine
    engine = ChunkEngine()

    print("✓ Chunk Engine initialized successfully\n")

    # Test Python file
    print("Testing Python file chunking...")
    python_file = create_sample_python_file()
    python_chunks = engine.chunk_file(python_file)
    print_chunk_summary(python_chunks, "Python")

    # Test Markdown file
    print("Testing Markdown file chunking...")
    markdown_file = create_sample_markdown_file()
    markdown_chunks = engine.chunk_file(markdown_file)
    print_chunk_summary(markdown_chunks, "Markdown")

    # Test JSON file
    print("Testing JSON file chunking...")
    json_file = create_sample_json_file()
    json_chunks = engine.chunk_file(json_file)
    print_chunk_summary(json_chunks, "JSON")

    # Test JavaScript file
    print("Testing JavaScript file chunking...")
    js_file = create_sample_javascript_file()
    js_chunks = engine.chunk_file(js_file)
    print_chunk_summary(js_chunks, "JavaScript")

    # Test batch processing
    print("Testing batch processing...")
    all_files = [python_file, markdown_file, json_file, js_file]
    all_chunks = engine.chunk_files(all_files)

    print(f"\n{'=' * 60}")
    print("Batch Processing Results")
    print(f"{'=' * 60}")
    print(f"Total files processed: {len(all_files)}")
    print(f"Total chunks generated: {len(all_chunks)}")
    print()

    # Group by chunk type
    from collections import Counter
    chunk_types = Counter(chunk.chunk_type.value for chunk in all_chunks)
    print("Chunks by type:")
    for chunk_type, count in chunk_types.items():
        print(f"  {chunk_type}: {count}")

    print("\n✓ All tests completed successfully!")


if __name__ == "__main__":
    main()
