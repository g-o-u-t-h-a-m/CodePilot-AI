"""Simple verification script for Chunk Engine."""

from app.chunking import ChunkEngine, initialize_strategies, CodeChunk, ChunkType
from app.models.source_file import SourceFile

# Initialize
initialize_strategies()
engine = ChunkEngine()

# Create a simple Python test file
python_content = '''def hello():
    print("Hello")

class MyClass:
    def method(self):
        pass
'''

source_file = SourceFile(
    path="/test/indexed_repos/test/example.py",
    relative_path="example.py",
    extension=".py",
    language="Python",
    size=len(python_content),
    line_count=len(python_content.splitlines()),
    encoding="utf-8",
    sha256="test",
    content=python_content
)

# Chunk it
chunks = engine.chunk_file(source_file)

print(f"SUCCESS: Generated {len(chunks)} chunks")
print(f"Chunk types: {[c.chunk_type.value for c in chunks]}")
print(f"Chunk names: {[c.metadata.get('function_name') or c.metadata.get('class_name') or 'N/A' for c in chunks]}")
