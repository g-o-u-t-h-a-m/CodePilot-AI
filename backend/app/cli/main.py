"""Command-line client for the CodePilot backend.

Sprint 10A. This module is a thin, presentation-only layer: it parses
arguments, calls the ``CodePilotClient``, and renders results as
developer-friendly ASCII text. It contains NO backend business logic and
never imports the core pipeline (RepositoryManager, RepositoryScanner,
ChunkEngine, EmbeddingEngine, ChromaDB, Retriever, PromptBuilder,
RAGService, LLMProvider). Everything is delegated over HTTP to the existing
FastAPI application.

Commands:
    health                          Check backend status
    clone <github-url>              Clone a repository (via the backend)
    index <repository-name>         Index an already-cloned repository
    ask <repository-name> "<question>"  Ask a question about an indexed repo

Exit codes:
    0  success
    1  operational/verification failure (a command failed, not the CLI)
    2  usage error (bad arguments)

The backend URL is configurable via the ``CODEPILOT_API_URL`` environment
variable (default ``http://127.0.0.1:8000``).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from app.cli.client import (
    API_URL_ENV,
    DEFAULT_BASE_URL,
    CodePilotClient,
    CodePilotConnectionError,
    CodePilotError,
    CodePilotHTTPError,
    CodePilotProtocolError,
)

# ASCII indicators (Windows cp1252 / PowerShell safe; no Unicode, no colors).
PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[INFO]"
ERROR = "[ERROR]"

# Exit codes.
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2

#: Header separator width used for answer output.
_SEP_WIDTH = 66


class _CliError(Exception):
    """Internal error carrying a pre-rendered message and exit code."""

    def __init__(self, message: str, exit_code: int = EXIT_ERROR):
        super().__init__(message)
        self.message = message
        self.exit_code = exit_code


def print_error(message: str) -> None:
    """Print an ASCII error banner to stderr."""
    print(f"\n{ERROR} {message}", file=sys.stderr)


def os_env_url() -> str:
    """Return the configured backend URL or the default."""
    return (os.environ.get(API_URL_ENV) or DEFAULT_BASE_URL).rstrip("/")


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def _cmd_health(client: CodePilotClient, args: argparse.Namespace) -> int:
    """Render the result of GET /health."""
    del args  # unused
    try:
        body = client.health()
    except CodePilotError as exc:
        raise _CliError(_render_error(exc, client.base_url))
    status = body.get("status", "unknown")
    service = body.get("service", "")
    print("CodePilot API")
    print(f"Status: {status}")
    if service:
        print(f"Service: {service}")
    if status == "healthy":
        print(f"{PASS} Backend is reachable and healthy")
        return EXIT_OK
    print(f"{FAIL} Backend reported a non-healthy status")
    return EXIT_ERROR


def _cmd_clone(client: CodePilotClient, args: argparse.Namespace) -> int:
    """Render the result of POST /repository/clone."""
    github_url = args.github_url
    if not github_url.strip():
        raise _CliError("GitHub URL must not be empty.", EXIT_USAGE)
    print(f"{INFO} Cloning {github_url}")
    try:
        body = client.clone_repository(github_url)
    except CodePilotError as exc:
        raise _CliError(_render_error(exc, client.base_url))

    success = body.get("success") is True
    repo_name = body.get("repository_name", "")
    local_path = body.get("local_path", "")
    message = body.get("message", "")

    if success:
        print(f"{PASS} Clone succeeded")
    else:
        print(f"{FAIL} Clone failed")
    if repo_name:
        print(f"Repository: {repo_name}")
    if local_path:
        print(f"Local path: {local_path}")
    if message:
        print(f"Message: {message}")
    return EXIT_OK if success else EXIT_ERROR


def _cmd_index(client: CodePilotClient, args: argparse.Namespace) -> int:
    """Render the result of POST /repository/index."""
    repository_name = args.repository_name.strip()
    if not repository_name:
        raise _CliError("Repository name must not be empty.", EXIT_USAGE)
    print(f"{INFO} Indexing repository '{repository_name}'")
    try:
        body = client.index_repository(repository_name)
    except CodePilotError as exc:
        raise _CliError(_render_error(exc, client.base_url))

    success = body.get("success") is True
    print(f"Repository: {body.get('repository_name', repository_name)}")
    print(f"Status: {'indexed' if success else 'failed'}")
    print(f"Files scanned: {body.get('files_scanned', '?')}")
    print(f"Chunks generated: {body.get('chunks_generated', '?')}")
    print(f"Embeddings generated: {body.get('embeddings_generated', '?')}")
    print(f"Vectors stored: {body.get('vectors_stored', '?')}")
    duration = body.get("duration")
    print(f"Duration: {f'{duration:.3f}s' if isinstance(duration, (int, float)) else '?'}")
    embedding_model = body.get("embedding_model")
    if embedding_model:
        print(f"Embedding model: {embedding_model}")
    print(f"Message: {body.get('message', '')}")
    if success:
        print(f"{PASS} Repository indexed successfully")
        return EXIT_OK
    print(f"{FAIL} Repository indexing failed")
    return EXIT_ERROR


def _cmd_ask(client: CodePilotClient, args: argparse.Namespace) -> int:
    """Render the result of POST /rag/query."""
    repository_name = args.repository_name.strip()
    question = args.question.strip()
    if not repository_name:
        raise _CliError("Repository name must not be empty.", EXIT_USAGE)
    if not question:
        raise _CliError("Question must not be empty.", EXIT_USAGE)
    top_k = args.top_k

    try:
        body = client.query(repository_name, question, top_k=top_k)
    except CodePilotError as exc:
        raise _CliError(_render_error(exc, client.base_url))

    answer = body.get("answer", "")
    sources = body.get("sources", [])
    insufficient = body.get("insufficient_context") is True

    print(f"{INFO} Asked about repository '{repository_name}'")

    print("=" * _SEP_WIDTH)
    print("CodePilot Answer")
    print("=" * _SEP_WIDTH)
    if answer:
        print(answer)

    if insufficient:
        print()
        print("No sufficiently relevant repository context was found.")
        print("The answer should not be interpreted as being grounded in the repository.")

    print()
    print("=" * _SEP_WIDTH)
    print("Sources")
    print("=" * _SEP_WIDTH)
    if sources:
        for index, source in enumerate(sources, start=1):
            file_path = source.get("file_path", "?")
            start_line = source.get("start_line")
            end_line = source.get("end_line")
            relevance = source.get("relevance_score")
            line_str = "?"
            if isinstance(start_line, int) and isinstance(end_line, int):
                line_str = f"{start_line}-{end_line}"
            rel_str = "?"
            if isinstance(relevance, (int, float)):
                rel_str = f"{relevance:.2f}"
            print(f"{index}. {file_path}")
            print(f"   Lines: {line_str}")
            print(f"   Relevance: {rel_str}")
    else:
        print("(none)")

    model_name = body.get("model_name")
    provider_name = body.get("provider_name")
    if model_name or provider_name:
        print()
        print(f"Model: {model_name or 'unknown'}  Provider: {provider_name or 'unknown'}")

    return EXIT_OK


# ---------------------------------------------------------------------------
# Error rendering
# ---------------------------------------------------------------------------

def _render_error(exc: CodePilotError, base_url: str) -> str:
    """Convert a client error into a human-readable message.

    Args:
        exc: The client-side exception.
        base_url: The base URL the client actually talked to, so error
            messages name the real configured endpoint.
    """
    if isinstance(exc, CodePilotConnectionError):
        reason = f" ({exc.reason})" if exc.reason else ""
        return (
            f"CodePilot API is not reachable at {base_url}"
            f"{reason}. Is the backend running? Start it with:\n"
            "    uvicorn main:app --reload --port 8000"
        )
    if isinstance(exc, CodePilotHTTPError):
        detail = exc.detail or f"HTTP {exc.status_code}"
        if exc.status_code == 404:
            return (
                f"{detail}\n"
                "The repository may not exist locally or may not have been "
                "indexed yet. Index it with:\n"
                "    python -m app.cli index <repository-name>"
            )
        if exc.status_code == 409:
            return (
                f"{detail}\n"
                "The repository directory is not a valid clone. Re-run:\n"
                "    python -m app.cli clone <github-url>"
            )
        if exc.status_code == 422:
            return f"Request rejected by the API (validation): {detail}"
        if exc.status_code == 500:
            return f"The backend reported an internal error: {detail}"
        return f"The backend returned HTTP {exc.status_code}: {detail}"
    if isinstance(exc, CodePilotProtocolError):
        return f"Unexpected response from the API: {exc}"
    return str(exc)


# ---------------------------------------------------------------------------
# Argument parsing and dispatch
# ---------------------------------------------------------------------------

def add_url_option(parser: argparse.ArgumentParser) -> None:
    """Register the ``--url`` override on a parser.

    ``SUPPRESS`` keeps the namespace key absent unless explicitly given, so
    the top-level parser and each subparser can share the option without the
    subparser's default clobbering a value the top-level parser already set.
    """
    parser.add_argument(
        "--url",
        dest="url",
        default=argparse.SUPPRESS,
        help=(
            "Backend URL override (default: CODEPILOT_API_URL or "
            f"{DEFAULT_BASE_URL})"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description=(
            "CodePilot command-line client. Talks to the local CodePilot "
            "FastAPI backend over HTTP (default http://127.0.0.1:8000; "
            "override with the CODEPILOT_API_URL environment variable)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_url_option(parser)
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    # health
    parser_health = subparsers.add_parser(
        "health", help="Check backend status (GET /health)"
    )
    add_url_option(parser_health)

    # clone
    parser_clone = subparsers.add_parser(
        "clone", help="Clone a repository through the backend"
    )
    add_url_option(parser_clone)
    parser_clone.add_argument("github_url", help="GitHub repository URL")

    # index
    parser_index = subparsers.add_parser(
        "index", help="Index an already-cloned repository"
    )
    add_url_option(parser_index)
    parser_index.add_argument("repository_name", help="Repository name to index")

    # ask
    parser_ask = subparsers.add_parser(
        "ask", help="Ask a question about an indexed repository"
    )
    add_url_option(parser_ask)
    parser_ask.add_argument("repository_name", help="Repository name to query")
    parser_ask.add_argument("question", help="Natural-language question")
    parser_ask.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Maximum number of sources to retrieve (default: backend default)",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point. Returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        parser.print_help(sys.stdout)
        return EXIT_USAGE

    # Validate top_k client-side so a bad value never reaches the network.
    if args.command == "ask" and args.top_k is not None:
        if args.top_k < 1 or args.top_k > 20:
            print_error(
                "top_k must be an integer between 1 and 20 "
                f"(got {args.top_k})."
            )
            return EXIT_USAGE

    client = CodePilotClient(base_url=getattr(args, "url", None))

    handlers = {
        "health": _cmd_health,
        "clone": _cmd_clone,
        "index": _cmd_index,
        "ask": _cmd_ask,
    }
    handler = handlers[args.command]

    try:
        return handler(client, args)
    except _CliError as exc:
        print_error(exc.message)
        return exc.exit_code
    except CodePilotError as exc:
        print_error(_render_error(exc, client.base_url))
        return EXIT_ERROR
    except KeyboardInterrupt:
        print_error("Interrupted.")
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
