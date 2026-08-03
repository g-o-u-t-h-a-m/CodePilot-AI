"""
Configuration constants for the repository scanner.
"""
from typing import Set

# Maximum file size to scan (2MB)
MAX_FILE_SIZE: int = 2 * 1024 * 1024

# Directories to ignore during scanning
IGNORED_DIRECTORIES: Set[str] = {
    ".git",
    "node_modules",
    "vendor",
    "dist",
    "build",
    "coverage",
    "target",
    ".idea",
    ".vscode",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "bin",
    "obj",
    "out"
}

# Binary and media file extensions to ignore
IGNORED_EXTENSIONS: Set[str] = {
    # Images
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".webp",
    # Videos
    ".mp4", ".avi", ".mov", ".wmv", ".flv", ".mkv",
    # Audio
    ".mp3", ".wav", ".flac", ".aac", ".ogg",
    # Documents
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    # Archives
    ".zip", ".tar", ".gz", ".rar", ".7z", ".bz2",
    # Executables
    ".exe", ".dll", ".so", ".dylib", ".bin",
    # Fonts
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
    # Other binary formats
    ".pyc", ".pyo", ".class", ".o", ".obj", ".jar", ".war"
}

# Extensionless files that should be scanned
SUPPORTED_EXTENSIONLESS: Set[str] = {
    "Dockerfile",
    "Makefile",
    "Jenkinsfile",
    "README",
    "LICENSE"
}
