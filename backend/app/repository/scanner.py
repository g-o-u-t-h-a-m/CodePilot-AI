import os
import hashlib
import logging
import chardet
from pathlib import Path
from typing import List, Optional
from app.models.source_file import SourceFile
from app.config import (
    MAX_FILE_SIZE,
    IGNORED_DIRECTORIES,
    IGNORED_EXTENSIONS,
    SUPPORTED_EXTENSIONLESS
)


logger = logging.getLogger(__name__)


class RepositoryScanner:
    """Scans repository directories and extracts source file information."""

    # Language detection mapping
    LANGUAGE_MAP: dict[str, str] = {
        ".py": "Python",
        ".js": "JavaScript",
        ".jsx": "JavaScript (JSX)",
        ".ts": "TypeScript",
        ".tsx": "TypeScript (TSX)",
        ".java": "Java",
        ".c": "C",
        ".cpp": "C++",
        ".cc": "C++",
        ".cxx": "C++",
        ".h": "C/C++ Header",
        ".hpp": "C++ Header",
        ".cs": "C#",
        ".go": "Go",
        ".rs": "Rust",
        ".rb": "Ruby",
        ".php": "PHP",
        ".swift": "Swift",
        ".kt": "Kotlin",
        ".scala": "Scala",
        ".r": "R",
        ".m": "Objective-C",
        ".sh": "Shell",
        ".bash": "Bash",
        ".zsh": "Zsh",
        ".ps1": "PowerShell",
        ".bat": "Batch",
        ".cmd": "Batch",
        ".html": "HTML",
        ".htm": "HTML",
        ".css": "CSS",
        ".scss": "SCSS",
        ".sass": "Sass",
        ".less": "Less",
        ".xml": "XML",
        ".json": "JSON",
        ".yaml": "YAML",
        ".yml": "YAML",
        ".toml": "TOML",
        ".ini": "INI",
        ".cfg": "Config",
        ".conf": "Config",
        ".md": "Markdown",
        ".txt": "Plain Text",
        ".sql": "SQL",
        ".graphql": "GraphQL",
        ".gql": "GraphQL",
        ".vue": "Vue",
        ".svelte": "Svelte",
        ".dart": "Dart",
        ".lua": "Lua",
        ".pl": "Perl",
        ".ex": "Elixir",
        ".exs": "Elixir",
        ".erl": "Erlang",
        ".hrl": "Erlang Header",
        ".clj": "Clojure",
        ".cljs": "ClojureScript",
        ".fs": "F#",
        ".fsx": "F#",
        ".ml": "OCaml",
        ".mli": "OCaml Interface"
    }

    def __init__(self):
        """Initialize the RepositoryScanner."""
        logger.info("RepositoryScanner initialized")

    def scan(self, repository_path: str) -> List[SourceFile]:
        """
        Scan a repository directory and extract source file information.

        Args:
            repository_path: Path to the repository root directory

        Returns:
            List of SourceFile objects

        Raises:
            ValueError: If repository path is invalid
            OSError: If there are filesystem access issues
        """
        repo_path = Path(repository_path)

        if not repo_path.exists():
            logger.error(f"Repository path does not exist: {repository_path}")
            raise ValueError(f"Repository path does not exist: {repository_path}")

        if not repo_path.is_dir():
            logger.error(f"Repository path is not a directory: {repository_path}")
            raise ValueError(f"Repository path is not a directory: {repository_path}")

        logger.info(f"Starting scan of repository: {repository_path}")
        source_files: List[SourceFile] = []

        try:
            for root, dirs, files in os.walk(repo_path):
                # Filter out ignored directories in-place
                dirs[:] = [d for d in dirs if not self.should_ignore_directory(d)]

                for filename in files:
                    file_path = Path(root) / filename

                    # Skip files that should be ignored
                    if self.should_ignore_file(file_path):
                        continue

                    try:
                        source_file = self._process_file(file_path, repo_path)
                        if source_file:
                            source_files.append(source_file)
                            logger.debug(f"Processed file: {source_file.relative_path}")
                    except Exception as e:
                        logger.warning(f"Failed to process file {file_path}: {e}")
                        continue

            logger.info(f"Scan completed. Found {len(source_files)} source files")
            return source_files

        except Exception as e:
            logger.error(f"Error during repository scan: {e}")
            raise

    def should_ignore_directory(self, directory_name: str) -> bool:
        """
        Check if a directory should be ignored during scanning.

        Args:
            directory_name: Name of the directory

        Returns:
            True if directory should be ignored, False otherwise
        """
        return directory_name in IGNORED_DIRECTORIES

    def should_ignore_file(self, file_path: Path) -> bool:
        """
        Check if a file should be ignored during scanning.

        Args:
            file_path: Path to the file

        Returns:
            True if file should be ignored, False otherwise
        """
        filename = file_path.name
        extension = file_path.suffix.lower()

        # Ignore files with binary/media extensions
        if extension in IGNORED_EXTENSIONS:
            return True

        # Ignore minified files
        if filename.endswith('.min.js') or filename.endswith('.min.css'):
            return True

        # Ignore hidden files (starting with .)
        if filename.startswith("."):
            return True

        # Allow supported extensionless files
        if not extension:
            if filename in SUPPORTED_EXTENSIONLESS:
                return False
            return True

        return False

    def detect_language(self, file_path: Path) -> str:
        """
        Detect programming language from file extension.

        Args:
            file_path: Path to the file

        Returns:
            Detected language name or "Unknown"
        """
        extension = file_path.suffix.lower()
        return self.LANGUAGE_MAP.get(extension, "Unknown")

    def calculate_sha256(self, content: str) -> str:
        """
        Calculate SHA-256 hash of file content.

        Args:
            content: File content as string

        Returns:
            Hexadecimal SHA-256 hash
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def read_file(self, file_path: Path) -> tuple[str, str]:
        """
        Read file content and detect encoding.

        Args:
            file_path: Path to the file

        Returns:
            Tuple of (content, encoding)

        Raises:
            OSError: If file cannot be read
            UnicodeDecodeError: If file cannot be decoded
        """
        # Read raw bytes first
        with open(file_path, "rb") as f:
            raw_data = f.read()

        # Detect encoding
        detected = chardet.detect(raw_data)
        encoding = detected.get("encoding", "utf-8")

        # Fallback to utf-8 if detection fails
        if not encoding or encoding.lower() == "ascii":
            encoding = "utf-8"

        try:
            # Decode content
            content = raw_data.decode(encoding)
            return content, encoding
        except UnicodeDecodeError:
            # Try with utf-8 as fallback
            try:
                content = raw_data.decode("utf-8")
                return content, "utf-8"
            except UnicodeDecodeError:
                # Last resort: decode with errors ignored
                content = raw_data.decode("utf-8", errors="ignore")
                logger.warning(f"File {file_path} decoded with errors ignored")
                return content, "utf-8"

    def _process_file(self, file_path: Path, repo_path: Path) -> Optional[SourceFile]:
        """
        Process a single file and create a SourceFile object.

        Args:
            file_path: Path to the file
            repo_path: Path to the repository root

        Returns:
            SourceFile object or None if processing fails
        """
        try:
            # Check file size before reading
            file_size = file_path.stat().st_size
            if file_size > MAX_FILE_SIZE:
                logger.info(f"Skipping oversized file: {file_path}")
                return None

            # Read file content and detect encoding
            content, encoding = self.read_file(file_path)

            # Calculate file metadata
            line_count = content.count("\n") + 1 if content else 0
            relative_path = file_path.relative_to(repo_path).as_posix()
            extension = file_path.suffix.lower()
            language = self.detect_language(file_path)
            sha256_hash = self.calculate_sha256(content)

            return SourceFile(
                path=str(file_path.absolute()),
                relative_path=relative_path,
                extension=extension,
                language=language,
                size=file_size,
                line_count=line_count,
                encoding=encoding,
                sha256=sha256_hash,
                content=content
            )

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return None
