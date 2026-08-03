# Sprint 4: Chunk Engine - Architecture Documentation

## Overview

The Chunk Engine is a flexible, extensible system for splitting source code files into meaningful chunks. It uses the **Strategy Pattern** and **Registry Pattern** to support multiple programming languages and file types while remaining open for extension.

---

## Folder Structure

```
app/chunking/
├── __init__.py                 # Module initialization and strategy registration
├── models.py                   # Data models (CodeChunk, ChunkType)
├── strategy.py                 # Abstract base class for strategies
├── registry.py                 # Strategy registry implementation
├── engine.py                   # Main chunk engine
└── strategies/                 # Strategy implementations
    ├── __init__.py
    ├── code/                   # Programming language strategies
    │   ├── __init__.py
    │   ├── python.py          # Python AST-based chunking
    │   ├── javascript.py      # JavaScript/TypeScript chunking
    │   ├── java.py            # Java chunking
    │   └── generic_code.py    # Generic code chunking (120 lines, 20 overlap)
    ├── docs/                   # Document strategies
    │   ├── __init__.py
    │   └── markdown.py        # Markdown heading-based chunking
    ├── config/                 # Configuration file strategies
    │   ├── __init__.py
    │   ├── json.py            # JSON whole-file chunking
    │   └── yaml.py            # YAML whole-file chunking
    └── generic/                # Fallback strategies
        ├── __init__.py
        └── generic.py         # Generic fallback (120 lines, 20 overlap)
```

---

## File Responsibilities

### Core Files

#### `models.py`
- **Responsibility**: Define data structures for chunks
- **Contents**:
  - `ChunkType` enum: FUNCTION, CLASS, MODULE, CONFIG, DOCUMENT, GENERIC
  - `CodeChunk` Pydantic model: Represents a single code chunk with metadata
- **Design Decision**: Using Pydantic ensures type safety and validation

#### `strategy.py`
- **Responsibility**: Define the contract for all chunking strategies
- **Contents**:
  - `ChunkStrategy` abstract base class with `chunk(source_file)` method
- **Design Decision**: Abstract base class enforces consistent interface across all strategies

#### `registry.py`
- **Responsibility**: Manage strategy registration and retrieval
- **Contents**:
  - `StrategyRegistry` class: Maps languages/extensions to strategies
  - Global registry instance with helper functions
- **Design Decision**: Centralized registry decouples engine from strategy implementations

#### `engine.py`
- **Responsibility**: Main entry point for chunking operations
- **Contents**:
  - `ChunkEngine` class: Orchestrates chunking process
  - Methods: `chunk_file()`, `chunk_files()`
- **Design Decision**: Engine knows nothing about specific strategies, only delegates to registry

### Strategy Files

#### `strategies/code/python.py`
- **Responsibility**: Parse and chunk Python files
- **Implementation**: Uses Python's `ast` module to extract functions and classes
- **Fallback**: Returns entire file as MODULE chunk if no functions/classes found

#### `strategies/code/javascript.py`
- **Responsibility**: Parse and chunk JavaScript/TypeScript files
- **Implementation**: Uses regex patterns to find functions, arrow functions, and classes
- **Fallback**: Returns entire file as MODULE chunk if no definitions found

#### `strategies/code/java.py`
- **Responsibility**: Parse and chunk Java files
- **Implementation**: Uses regex patterns to find classes, interfaces, and methods
- **Fallback**: Returns entire file as MODULE chunk if no definitions found

#### `strategies/code/generic_code.py`
- **Responsibility**: Chunk unsupported programming languages
- **Implementation**: Fixed-size chunks (120 lines) with overlap (20 lines)
- **Use Case**: Languages without dedicated parsers (C, C++, Go, Rust, etc.)

#### `strategies/docs/markdown.py`
- **Responsibility**: Chunk Markdown documentation
- **Implementation**: Splits by heading markers (`#`, `##`, etc.)
- **Fallback**: Returns entire file as one DOCUMENT chunk if no headings found

#### `strategies/config/json.py` & `yaml.py`
- **Responsibility**: Handle configuration files
- **Implementation**: Treats entire file as single CONFIG chunk
- **Rationale**: Config files are typically small and semantically cohesive

#### `strategies/generic/generic.py`
- **Responsibility**: Fallback for completely unknown file types
- **Implementation**: Fixed-size chunks (120 lines) with overlap (20 lines)
- **Use Case**: Files that don't match any registered strategy

---

## Design Patterns

### Why Strategy Pattern?

The **Strategy Pattern** was chosen because:

1. **Encapsulation**: Each language's chunking logic is encapsulated in its own class
2. **Interchangeability**: Strategies can be swapped at runtime based on file type
3. **Testability**: Each strategy can be tested independently
4. **Single Responsibility**: Each strategy has one job - chunk its specific file type

**Example**: Python uses AST parsing, JavaScript uses regex, Markdown splits by headings - completely different algorithms, same interface.

### Why Registry Pattern?

The **Registry Pattern** was chosen because:

1. **Open/Closed Principle**: New strategies can be added without modifying existing code
2. **Decoupling**: ChunkEngine doesn't know about specific strategies
3. **Flexibility**: Multiple keys (language name, file extension) can map to same strategy
4. **Centralized Management**: Single point of truth for strategy mappings

**Example**: Both "python" and ".py" map to `PythonChunkStrategy` without duplicating code.

---

## Adding a New Language (Example: Rust)

To add support for Rust **without modifying ChunkEngine**, follow these steps:

### Step 1: Create the Strategy

Create `app/chunking/strategies/code/rust.py`:

```python
"""Rust code chunking strategy."""

import hashlib
import logging
import re
from typing import List

from app.chunking.strategy import ChunkStrategy
from app.chunking.models import CodeChunk, ChunkType
from app.models.source_file import SourceFile

logger = logging.getLogger(__name__)


class RustChunkStrategy(ChunkStrategy):
    """Chunking strategy for Rust files."""

    FUNCTION_PATTERN = re.compile(r'^\s*(?:pub\s+)?fn\s+(\w+)', re.MULTILINE)
    STRUCT_PATTERN = re.compile(r'^\s*(?:pub\s+)?struct\s+(\w+)', re.MULTILINE)
    IMPL_PATTERN = re.compile(r'^\s*impl\s+(\w+)', re.MULTILINE)

    def chunk(self, source_file: SourceFile) -> List[CodeChunk]:
        """Split Rust source file into chunks."""
        # Implementation similar to JavaChunkStrategy
        # ... (extract functions, structs, impl blocks)
        pass
```

### Step 2: Register the Strategy

Add to `app/chunking/__init__.py` in the `initialize_strategies()` function:

```python
from app.chunking.strategies.code.rust import RustChunkStrategy

def initialize_strategies():
    # ... existing registrations ...
    
    # Register Rust strategy
    register_strategy("rust", RustChunkStrategy)
    register_strategy(".rs", RustChunkStrategy)
```

### Step 3: Export from Module

Add to `app/chunking/strategies/code/__init__.py`:

```python
from app.chunking.strategies.code.rust import RustChunkStrategy

__all__ = [
    # ... existing exports ...
    "RustChunkStrategy",
]
```

### That's It!

**No changes required to:**
- `ChunkEngine` class
- `StrategyRegistry` class
- Any existing strategies
- Any other core files

The engine automatically uses `RustChunkStrategy` for all Rust files.

---

## Architecture Principles

### SOLID Compliance

1. **Single Responsibility**: Each strategy handles one file type
2. **Open/Closed**: Open for extension (add strategies), closed for modification (engine unchanged)
3. **Liskov Substitution**: All strategies implement `ChunkStrategy` interface
4. **Interface Segregation**: Simple single-method interface (`chunk()`)
5. **Dependency Inversion**: Engine depends on abstract `ChunkStrategy`, not concrete implementations

### Loose Coupling

- **ChunkEngine** → knows about → **StrategyRegistry** (not specific strategies)
- **StrategyRegistry** → knows about → **ChunkStrategy** interface (not implementations)
- **Strategies** → know about → `SourceFile` and `CodeChunk` models (not engine)

### Scalability

The architecture scales in multiple dimensions:

1. **Languages**: Add new programming languages without limit
2. **File Types**: Add document types, config formats, etc.
3. **Strategy Complexity**: Simple (regex) or complex (AST parsing)
4. **Chunk Granularity**: Fine-grained (functions) or coarse (modules)

### Production Quality

- ✅ Full type hints
- ✅ Comprehensive logging
- ✅ Error handling with fallbacks
- ✅ Content hashing for deduplication
- ✅ Metadata for downstream processing
- ✅ Batch processing support
- ❌ No placeholder code
- ❌ No TODO comments
- ❌ No integration with embeddings/RAG (as specified)

---

## Usage Example

```python
from app.chunking import ChunkEngine, initialize_strategies
from app.models.source_file import SourceFile

# Initialize once at application startup
initialize_strategies()

# Create engine
engine = ChunkEngine()

# Chunk a single file
chunks = engine.chunk_file(source_file)

# Chunk multiple files
all_chunks = engine.chunk_files([file1, file2, file3])

# Process chunks
for chunk in chunks:
    print(f"{chunk.chunk_type}: {chunk.relative_path}")
    print(f"  Lines {chunk.start_line}-{chunk.end_line}")
    print(f"  Hash: {chunk.content_hash}")
    print(f"  Metadata: {chunk.metadata}")
```

---

## Key Benefits

1. **Maintainability**: Each component has clear responsibility
2. **Extensibility**: Add languages without modifying existing code
3. **Testability**: Strategies can be unit tested independently
4. **Flexibility**: Different chunking approaches for different file types
5. **Reliability**: Fallback strategies prevent failures on unknown files

---

## Future Extensions

The architecture supports future enhancements without breaking changes:

- **Custom Chunk Sizes**: Pass parameters to strategy constructors
- **Chunk Filtering**: Add filters to exclude certain chunk types
- **Parallel Processing**: Chunk multiple files concurrently
- **Chunk Validation**: Add validators to ensure chunk quality
- **Strategy Metrics**: Track performance of each strategy
- **Dynamic Registration**: Load strategies from plugins at runtime

All of these can be added **without modifying the core engine or existing strategies**.
