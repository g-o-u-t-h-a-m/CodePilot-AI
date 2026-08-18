"""Entry point for ``python -m app.cli``.

Delegates to the command layer in ``app.cli.main`` so the same logic is
usable both as ``python -m app.cli ...`` and programmatically via
``app.cli.main.main``.
"""

import sys

from app.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
