"""
Comprehensive test to verify all Sprint 3 improvements.
"""
import logging
import tempfile
from pathlib import Path
from app.repository.scanner import RepositoryScanner
from app.config import MAX_FILE_SIZE

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_improvements():
    """Test all Sprint 3 improvements."""
    scanner = RepositoryScanner()

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)

        # Create test files
        test_files = {
            'Dockerfile': 'FROM python:3.11\nRUN pip install fastapi',
            'Makefile': 'build:\n\tpython setup.py build',
            'Jenkinsfile': 'pipeline { agent any }',
            'README': '# Test Project\nThis is a test.',
            'LICENSE': 'MIT License\nCopyright 2026',
            'script.js': 'console.log("hello");',
            'app.min.js': 'console.log("minified");',
            'styles.min.css': '.btn{color:red}',
            'main.py': 'print("hello")',
            '.hidden': 'should be ignored',
            'image.png': b'\x89PNG\r\n\x1a\n',  # Binary file
        }

        # Create oversized file (> 2MB)
        oversized_content = 'x' * (MAX_FILE_SIZE + 1000)

        logger.info("Creating test files...")
        for filename, content in test_files.items():
            file_path = repo_path / filename
            if isinstance(content, bytes):
                file_path.write_bytes(content)
            else:
                file_path.write_text(content)

        # Create oversized file
        (repo_path / 'oversized.txt').write_text(oversized_content)

        logger.info(f"Scanning test repository: {repo_path}")
        results = scanner.scan(str(repo_path))

        # Analyze results
        scanned_files = {f.relative_path for f in results}

        logger.info("\n" + "="*60)
        logger.info("TEST RESULTS")
        logger.info("="*60)

        # Test 1: Extensionless files should be scanned
        extensionless = {'Dockerfile', 'Makefile', 'Jenkinsfile', 'README', 'LICENSE'}
        found_extensionless = scanned_files & extensionless
        logger.info(f"\n1. Extensionless files scanned: {found_extensionless}")
        assert found_extensionless == extensionless, f"Expected {extensionless}, got {found_extensionless}"
        logger.info("   ✓ PASS: All extensionless files scanned")

        # Test 2: Minified files should be ignored
        assert 'app.min.js' not in scanned_files, "Minified JS should be ignored"
        assert 'styles.min.css' not in scanned_files, "Minified CSS should be ignored"
        logger.info("\n2. Minified files ignored:")
        logger.info("   ✓ PASS: app.min.js ignored")
        logger.info("   ✓ PASS: styles.min.css ignored")

        # Test 3: Regular source files should be scanned
        assert 'script.js' in scanned_files, "Regular JS should be scanned"
        assert 'main.py' in scanned_files, "Python files should be scanned"
        logger.info("\n3. Regular source files scanned:")
        logger.info("   ✓ PASS: script.js scanned")
        logger.info("   ✓ PASS: main.py scanned")

        # Test 4: Hidden files should be ignored
        assert '.hidden' not in scanned_files, "Hidden files should be ignored"
        logger.info("\n4. Hidden files ignored:")
        logger.info("   ✓ PASS: .hidden ignored")

        # Test 5: Binary files should be ignored
        assert 'image.png' not in scanned_files, "Binary files should be ignored"
        logger.info("\n5. Binary files ignored:")
        logger.info("   ✓ PASS: image.png ignored")

        # Test 6: Oversized files should be skipped
        assert 'oversized.txt' not in scanned_files, "Oversized files should be skipped"
        logger.info("\n6. Oversized files skipped:")
        logger.info(f"   ✓ PASS: oversized.txt ({MAX_FILE_SIZE + 1000} bytes) skipped")

        logger.info("\n" + "="*60)
        logger.info("ALL TESTS PASSED!")
        logger.info("="*60)
        logger.info(f"\nTotal files scanned: {len(results)}")
        logger.info(f"Files: {sorted(scanned_files)}")

        return True


if __name__ == "__main__":
    try:
        test_improvements()
    except AssertionError as e:
        logger.error(f"TEST FAILED: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"ERROR: {e}", exc_info=True)
        exit(1)
