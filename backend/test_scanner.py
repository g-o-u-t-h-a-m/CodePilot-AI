"""
Test script to verify RepositoryScanner implementation.
"""
import logging
from pathlib import Path
from app.repository.scanner import RepositoryScanner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Test the RepositoryScanner on the current project."""
    scanner = RepositoryScanner()

    # Test on the backend directory itself
    backend_path = Path(__file__).parent
    logger.info(f"Scanning directory: {backend_path}")

    try:
        source_files = scanner.scan(str(backend_path))

        logger.info(f"\n{'='*60}")
        logger.info(f"Scan Results:")
        logger.info(f"{'='*60}")
        logger.info(f"Total files scanned: {len(source_files)}")

        # Group by language
        languages = {}
        for file in source_files:
            languages[file.language] = languages.get(file.language, 0) + 1

        logger.info(f"\nFiles by language:")
        for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {lang}: {count}")

        # Show first 5 files as examples
        logger.info(f"\nFirst 5 files:")
        for file in source_files[:5]:
            logger.info(f"\n  Path: {file.relative_path}")
            logger.info(f"  Language: {file.language}")
            logger.info(f"  Size: {file.size} bytes")
            logger.info(f"  Lines: {file.line_count}")
            logger.info(f"  Encoding: {file.encoding}")
            logger.info(f"  SHA256: {file.sha256[:16]}...")

        logger.info(f"\n{'='*60}")
        logger.info("✓ Scanner test completed successfully!")
        logger.info(f"{'='*60}")

    except Exception as e:
        logger.error(f"Error during scan: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
