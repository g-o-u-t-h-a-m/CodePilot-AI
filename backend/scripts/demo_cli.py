"""End-to-end CLI verification for Sprint 10A (command-line client).

This script verifies the REAL CLI/client boundary. Unlike earlier demos it
does not call internal services: it starts the actual FastAPI application in
a separate uvicorn process (on a throwaway port, from a temporary sandbox so
repositories and ChromaDB never touch the real project), then runs
``python -m app.cli ...`` as a subprocess over that real HTTP connection.

Verifications cover Part 10 of the sprint spec:
    1.  health command
    2.  clone command using a local/test-safe mechanism (no GitHub / network)
    3.  index command displays repository, files, chunks, embeddings, vectors
    4.  ask command returns answer, repository name, sources, paths, line ranges
    5.  insufficient context communicated clearly
    6.  validation errors handled cleanly (no tracebacks)
    7.  nonexistent repository -> human-readable 404
    8.  backend unavailable -> connection failure without traceback
    9.  top_k: valid values work, invalid values rejected
    10. CODEPILOT_API_URL is respected (and --url override)
    11. no secrets / generated artifacts are added to the repository

Extra client-boundary checks with a stub HTTP server (deterministic, no
backend needed):
    - Exact request payloads (clone/index/ask bodies) transmitted over HTTP
    - top_k is transmitted when supplied and absent when not
    - malformed JSON responses are handled without a traceback

Requirements enforced by design:
    - No GitHub, no external network, no paid/cloud/OpenAI/Gemini/OpenRouter.
    - The mock LLM stays the default (no LLM_PROVIDER / LLM_API_KEY set).
    - The password is ASCII-only ([PASS]/[FAIL]/[ERROR]/[INFO]) so it runs
      correctly in Windows PowerShell / cp1252 terminals and needs no color
      library.
    - Exit code is 0 only when every check passes.

Run from the backend directory:
    python scripts/demo_cli.py
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Ignore the backend app INFO logging completely: this script talks to the
# backend over HTTP only and never imports app.* into this process.
logging.basicConfig(level=logging.WARNING)

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent
PYTHON = sys.executable

# Make ``app.cli`` importable for the URL-constants introspection checks
# (DEFAULT_BASE_URL / API_URL_ENV). Nothing from the core pipeline is imported.
sys.path.insert(0, str(BACKEND_DIR))

PASS = "[PASS]"
FAIL = "[FAIL]"
ERROR = "[ERROR]"

# Paths the demo itself is allowed to have created (verified against the
# repo's git state at the end).
EXPECTED_NEW_PATHS = {
    "backend/app/cli/__init__.py",
    "backend/app/cli/client.py",
    "backend/app/cli/main.py",
    "backend/app/cli/__main__.py",
    "backend/scripts/demo_cli.py",
    "backend/README.md",
}

SERVER_PROC: subprocess.Popen = None
SERVER_URL = ""
SERVER_LOG: Path = None

_checks_run = 0
_checks_passed = 0


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def check(condition: bool, label: str, detail: str = "") -> bool:
    """Record and print an ASCII verification verdict."""
    global _checks_run, _checks_passed
    _checks_run += 1
    if condition:
        _checks_passed += 1
        print(f"{PASS} {label}")
        return True
    print(f"{FAIL} {label}")
    if detail:
        print(f"     {detail}")
    return False


def section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'-' * 66}")
    print(f" {title}")
    print("-" * 66)


def final_verdict() -> None:
    """Summarize results; exit 0 only when every check passed."""
    print(f"\n{'=' * 66}")
    print(f" {_checks_passed}/{_checks_run} checks passed (Sprint 10A CLI)")
    print("=" * 66)
    if _checks_run > 0 and _checks_passed == _checks_run:
        print(f"{PASS} ALL VERIFICATIONS PASSED (exit code 0)")
        sys.exit(0)
    print(f"{FAIL} SOME VERIFICATIONS FAILED (exit code 1)")
    sys.exit(1)


def free_port() -> int:
    """Return a currently-free TCP port on 127.0.0.1."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def run_cli(args, env_extra=None, cwd=BACKEND_DIR, timeout=240):
    """Run the real CLI as a subprocess and capture its output.

    CODEPILOT_API_URL is cleared first so explicit test configuration is
    deterministic (unless the caller overrides it through ``env_extra``).

    Returns:
        A (returncode, stdout, stderr) tuple.
    """
    env = os.environ.copy()
    env.pop("CODEPILOT_API_URL", None)
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(
        [PYTHON, "-m", "app.cli", *args],
        capture_output=True, text=True, cwd=str(cwd), env=env,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr


def run_git(args, cwd=REPO_ROOT):
    """Run a git command and return trimmed stdout lines."""
    proc = subprocess.run(
        ["git", *args], capture_output=True, text=True, cwd=str(cwd),
    )
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def git_porcelain() -> set:
    """Return the set of changed/new repo paths (status prefixes stripped)."""
    paths = set()
    for line in run_git(["status", "--porcelain"]):
        # Strip the two-character status column (e.g. '?? ', ' M ', 'A  ').
        paths.add(line[3:].strip())
    return paths


def count_source_lines(stdout: str) -> int:
    """Count the numbered source entries rendered by the CLI."""
    return len(re.findall(r"^\d+\.\s", stdout, flags=re.MULTILINE))


def extract_field(stdout: str, label: str) -> str:
    """Extract the value of a 'Label: value' line from CLI output."""
    match = re.search(rf"^{re.escape(label)}:\s*(.*)$", stdout, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def int_field(stdout: str, label: str, default: int = -1) -> int:
    match = re.search(rf"^{re.escape(label)}:\s*(\d+)\s*$", stdout, flags=re.MULTILINE)
    return int(match.group(1)) if match else default


# ---------------------------------------------------------------------------
# Fixtures: small local repositories (no network)
# ---------------------------------------------------------------------------

REPO_A = "demo-repository"

REPO_A_CONTENTS = {
    "src/auth.py": (
        "def authenticate_user(username, password):\n"
        '    """Validate a username/password pair."""\n'
        "    user = users_db.get(username)\n"
        '    if user and user.check_password(password):\n'
        "        return user\n"
        "    return None\n"
    ),
    "src/db.py": (
        "def connect_database():\n"
        '    """Open the application database connection."""\n'
        "    import sqlite3\n"
        '    conn = sqlite3.connect("app.db")\n'
        "    return conn\n"
    ),
    "src/api/routes.py": (
        "def login_route(request):\n"
        '    """Handle POST /login."""\n'
        "    user = authenticate_user(request.username, request.password)\n"
        "    if user is None:\n"
        '        return error("invalid credentials")\n'
        "    return ok(session_token_for(user))\n"
    ),
    "src/utils/helpers.py": (
        "def tokenize(text):\n"
        '    """Split text into tokens."""\n'
        "    return text.split()\n"
    ),
}

AUTH_QUESTION = "Where is user authentication handled?"
INSUFFICIENT_QUESTION = "Where is blockchain mining implemented?"


def write_repo(scratch: Path, name: str, files: dict) -> None:
    """Write a small local git repository (RepositoryManager layout)."""
    from git import Repo

    source = scratch / "indexed_repos" / name / "source"
    source.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        path = source / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    Repo.init(str(source))


# ---------------------------------------------------------------------------
# Backend server process (real uvicorn, real network)
# ---------------------------------------------------------------------------

def start_backend(scratch: Path) -> str:
    """Start the real FastAPI backend in a subprocess from a sandbox.

    The server CWD is the sandbox, so relative paths (``indexed_repos/``)
    resolve there, and ChromaDB is redirected to ``<scratch>/chroma`` via
    CHROMA_PERSIST_DIR. The mock LLM provider stays the default.

    Returns:
        Base URL of the running server, e.g. ``http://127.0.0.1:12345``.
    """
    global SERVER_PROC, SERVER_URL, SERVER_LOG
    port = free_port()
    SERVER_URL = f"http://127.0.0.1:{port}"
    SERVER_LOG = scratch / "server.log"

    env = os.environ.copy()
    env["CHROMA_PERSIST_DIR"] = str(scratch / "chroma")
    env["COLLECTION_NAME"] = "sprint10a_cli_demo"
    env.pop("LLM_PROVIDER", None)
    env.pop("LLM_API_KEY", None)
    env["PYTHONPATH"] = str(BACKEND_DIR) + os.pathsep + env.get("PYTHONPATH", "")

    with open(SERVER_LOG, "w", encoding="utf-8") as log:
        SERVER_PROC = subprocess.Popen(
            [
                PYTHON, "-m", "uvicorn", "main:app",
                "--app-dir", str(BACKEND_DIR),
                "--host", "127.0.0.1",
                "--port", str(port),
                "--log-level", "warning",
            ],
            cwd=str(scratch),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return SERVER_URL


def wait_backend_ready(timeout: float = 180.0) -> bool:
    """Poll GET /health until the server responds or the deadline passes."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if SERVER_PROC.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(SERVER_URL + "/health", timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def stop_backend() -> None:
    """Terminate the backend subprocess and wait for it to exit."""
    global SERVER_PROC
    if SERVER_PROC is not None:
        SERVER_PROC.terminate()
        try:
            SERVER_PROC.wait(timeout=20)
        except subprocess.TimeoutExpired:
            SERVER_PROC.kill()
            SERVER_PROC.wait(timeout=10)
        SERVER_PROC = None


# ---------------------------------------------------------------------------
# Stub HTTP server (deterministic client-boundary checks)
# ---------------------------------------------------------------------------

# Shared state for the stub (per test run).
STUB_STATE = {"next_body": "{}", "requests": [], "thread": None, "httpd": None}


class _StubHandler(BaseHTTPRequestHandler):
    """Records every request and returns a scripted JSON body."""

    def _respond(self, code: int) -> None:
        body = STUB_STATE["next_body"].encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._respond(200)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length).decode("utf-8", "replace")
        try:
            payload = json.loads(raw) if raw.strip() else None
        except json.JSONDecodeError:
            payload = None
        STUB_STATE["requests"].append((self.path, payload))
        self._respond(200)

    def log_message(self, *args):  # silence request logging
        pass


def stub_url() -> str:
    """Start the stub HTTP server and return its base URL."""
    port = free_port()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _StubHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    STUB_STATE["httpd"] = httpd
    STUB_STATE["thread"] = thread
    STUB_STATE["requests"] = []
    return f"http://127.0.0.1:{port}"


def stub_stop() -> None:
    if STUB_STATE["httpd"] is not None:
        STUB_STATE["httpd"].shutdown()
        STUB_STATE["httpd"].server_close()
        STUB_STATE["thread"].join(timeout=5)
    STUB_STATE["httpd"] = None


def last_stub_request():
    return STUB_STATE["requests"][-1] if STUB_STATE["requests"] else (None, None)


# ---------------------------------------------------------------------------
# Verification sections
# ---------------------------------------------------------------------------

def verify_health_command() -> None:
    """Part 1: health command."""
    section("1. health command")
    code, out, err = run_cli(["health"], env_extra={"CODEPILOT_API_URL": SERVER_URL})
    check(code == 0, "1.1 health exits 0", f"exit={code} stderr={err[:120]}")
    check("CodePilot API" in out, "1.2 Output identifies CodePilot API")
    check("Status: healthy" in out, "1.3 Status reported as healthy", out)
    check("Service: CodePilot AI" in out,
          "1.4 Service name reported as 'CodePilot AI'", out)
    check(f"{PASS}" in out, "1.5 Success indicator printed", out)


def verify_clone_command() -> None:
    """Part 2: clone command using a local/test-safe mechanism.

    The repository already exists locally in the server sandbox, so the
    backend's clone endpoint short-circuits to success (no GitHub/network).
    A second run with an invalid URL exercises the failure rendering path.
    """
    section("2. clone command (local-safe, no GitHub/network)")
    url = f"https://github.com/codepilot-fixture/{REPO_A}"
    code, out, err = run_cli(["clone", url],
                             env_extra={"CODEPILOT_API_URL": SERVER_URL})
    check(code == 0, "2.1 clone of an existing local repo exits 0",
          f"exit={code} stderr={err[:120]}")
    check(f"{PASS} Clone succeeded" in out, "2.2 Success displayed", out)
    check(f"Repository: {REPO_A}" in out, "2.3 Repository name displayed")
    check("Local path:" in out, "2.4 Local path displayed")
    check("indexed_repos" in out, "2.5 Local path points into indexed_repos")
    check("Message: Repository already exists" in out,
          "2.6 Server message displayed", out)

    code, out, err = run_cli(["clone", "not-a-github-url"],
                             env_extra={"CODEPILOT_API_URL": SERVER_URL})
    check(code == 1, "2.7 clone with invalid URL exits 1",
          f"exit={code} stderr={err[:120]}")
    check(f"{FAIL} Clone failed" in out, "2.8 Clone failure displayed")
    check("Message: Invalid GitHub URL format" in out,
          "2.9 Failure message from the server displayed", out)


def verify_index_command() -> None:
    """Part 3: index command displays the structured statistics."""
    section("3. index command")
    code, out, err = run_cli(["index", REPO_A],
                             env_extra={"CODEPILOT_API_URL": SERVER_URL})
    check(code == 0, "3.1 index exits 0", f"exit={code} stderr={err[:120]}")
    if code != 0:
        print(f"     stdout: {out[:250]}")
        print(f"     stderr: {err[:250]}")
        return
    check(f"Repository: {REPO_A}" in out, "3.2 Repository name displayed")
    check("Status: indexed" in out, "3.3 Status shows indexed", out)
    files_scanned = int_field(out, "Files scanned")
    chunks = int_field(out, "Chunks generated")
    embeddings = int_field(out, "Embeddings generated")
    vectors = int_field(out, "Vectors stored")
    check(files_scanned == 4, "3.4 Files scanned reported as 4",
          f"got {files_scanned}")
    check(chunks > 0, "3.5 Chunks generated reported as a positive number",
          f"got {chunks}")
    check(embeddings == chunks and chunks > 0,
          "3.6 Embeddings generated matches chunks",
          f"chunks={chunks} embeddings={embeddings}")
    check(vectors == chunks and chunks > 0,
          "3.7 Vectors stored matches chunks", f"chunks={chunks} vectors={vectors}")
    check(bool(extract_field(out, "Duration")), "3.8 Duration reported")
    check("Message:" in out, "3.9 Server message displayed")
    check(f"{PASS} Repository indexed successfully" in out,
          "3.10 Success indicator printed")


def verify_index_errors() -> None:
    """Part 7 + 6: nonexistent repository index, empty arguments."""
    section("4. index errors (404 + validation)")
    code, out, err = run_cli(["index", "ghost-repository"],
                             env_extra={"CODEPILOT_API_URL": SERVER_URL})
    check(code == 1, "4.1 index of a nonexistent repository exits 1",
          f"exit={code}")
    check(f"{ERROR}" in err, "4.2 [ERROR] banner shown", err[:120])
    check("not present" in err, "4.3 404 explains the repository is missing", err)
    check("Traceback" not in err, "4.4 No Python traceback shown")

    code, out, err = run_cli(["index", ""],
                             env_extra={"CODEPILOT_API_URL": SERVER_URL})
    check(code == 2, "4.5 index with an empty name exits 2 (usage)",
          f"exit={code}")


def verify_ask_command() -> None:
    """Part 4: ask command returns a grounded answer with sources."""
    section(f"5. ask command ({AUTH_QUESTION!r})")
    code, out, err = run_cli(
        ["ask", REPO_A, AUTH_QUESTION],
        env_extra={"CODEPILOT_API_URL": SERVER_URL},
    )
    check(code == 0, "5.1 ask exits 0", f"exit={code} stderr={err[:120]}")
    if code != 0:
        print(f"     out: {out[:250]}")
        print(f"     err: {err[:250]}")
        return
    check("CodePilot Answer" in out, "5.2 Answer section header shown")
    check(AUTH_QUESTION in out, "5.3 The question is echoed", out[:200])
    check(REPO_A in out, "5.4 Repository name appears", out[:300])
    check("Sources" in out, "5.5 Sources section header shown")
    src_count = count_source_lines(out)
    check(src_count > 0, "5.6 At least one source is listed",
          f"count={src_count}")
    check("src/auth.py" in out,
          "5.7 Auth source file path is displayed", out[:400])
    check("Lines:" in out, "5.8 Line ranges are displayed")
    check("Relevance:" in out, "5.9 Relevance scores are displayed")
    check("Model: mock-llm" in out, "5.10 Model metadata is displayed concisely")
    check("Provider: mock" in out, "5.11 Provider metadata is displayed")
    check("embedding" not in out.lower() and "chroma" not in out.lower(),
          "5.12 No raw embedding vectors or ChromaDB info exposed")


def verify_insufficient_context() -> None:
    """Part 5: insufficient-context questions communicated clearly."""
    section(f"6. insufficient context ({INSUFFICIENT_QUESTION!r})")
    code, out, err = run_cli(
        ["ask", REPO_A, INSUFFICIENT_QUESTION],
        env_extra={"CODEPILOT_API_URL": SERVER_URL},
    )
    check(code == 0, "6.1 Insufficient-context ask exits 0 (HTTP 200)",
          f"exit={code} stderr={err[:120]}")
    check("No sufficiently relevant repository context was found." in out,
          "6.2 Insufficient-context notice displayed", out)
    check("not be interpreted as being grounded in the repository" in out,
          "6.3 Grounding caveat displayed")


def verify_nonexistent_repository_ask() -> None:
    """Part 7: nonexistent repository -> human-readable 404 (no traceback)."""
    section("7. ask against a nonexistent repository (404)")
    code, out, err = run_cli(
        ["ask", "ghost-repository", AUTH_QUESTION],
        env_extra={"CODEPILOT_API_URL": SERVER_URL},
    )
    check(code == 1, "7.1 Nonexistent repository ask exits 1", f"exit={code}")
    check(f"{ERROR}" in err, "7.2 [ERROR] banner shown", err[:160])
    check("has not been indexed" in err, "7.3 404 explains the failure", err)
    check("Traceback" not in err, "7.4 No Python traceback shown")


def verify_top_k() -> None:
    """Part 9: valid top_k works, invalid top_k is rejected."""
    section("8. top_k handling")
    code, out, err = run_cli(
        ["ask", REPO_A, AUTH_QUESTION, "--top-k", "1"],
        env_extra={"CODEPILOT_API_URL": SERVER_URL},
    )
    check(code == 0, "8.1 ask --top-k 1 exits 0", f"exit={code} stderr={err[:120]}")
    src_count = count_source_lines(out)
    check(0 <= src_count <= 1, "8.2 top_k=1 caps the source list",
          f"sources={src_count}")

    code, out, err = run_cli(
        ["ask", REPO_A, AUTH_QUESTION, "--top-k", "20"],
        env_extra={"CODEPILOT_API_URL": SERVER_URL},
    )
    check(code == 0, "8.3 ask --top-k 20 exits 0", f"exit={code} stderr={err[:120]}")

    for bad in ("0", "-3", "21"):
        code, out, err = run_cli(
            ["ask", REPO_A, AUTH_QUESTION, "--top-k", bad],
            env_extra={"CODEPILOT_API_URL": SERVER_URL},
        )
        check(code == 2, f"8.4 ask --top-k {bad} is rejected (exit 2)",
              f"exit={code}")
        check(f"{ERROR}" in err, f"8.5 --top-k {bad} prints [ERROR]", err[:120])
        check("Traceback" not in err, f"8.6 --top-k {bad} shows no traceback")


def verify_validation_errors() -> None:
    """Part 6: malformed/empty arguments are handled cleanly."""
    section("9. overall CLI validation")
    code, out, err = run_cli([])
    check(code == 2, "9.1 No command prints usage and exits 2", f"exit={code}")
    check("usage:" in out + err, "9.2 Usage text shown")

    code, out, err = run_cli(["ask", REPO_A, ""],
                             env_extra={"CODEPILOT_API_URL": SERVER_URL})
    check(code == 2, "9.3 ask with an empty question exits 2", f"exit={code}")
    check(f"{ERROR}" in err, "9.4 Empty question prints [ERROR]", err[:120])


def verify_url_configuration() -> None:
    """Part 10: CODEPILOT_API_URL is respected (and --url override)."""
    section("10. server URL configuration")

    # COD EPILOT API URL env var -> real server.
    code, out, err = run_cli(["health"], env_extra={"CODEPILOT_API_URL": SERVER_URL})
    check(code == 0 and f"{PASS}" in out,
          "10.1 CODEPILOT_API_URL routes health to the test server",
          f"exit={code}")

    # --url override (no env var set) -> real server.
    code, out, err = run_cli(["health", "--url", SERVER_URL])
    check(code == 0 and f"{PASS}" in out,
          "10.2 --url override routes health to the test server",
          f"exit={code}")

    # Client default constant is the documented default.
    from app.cli.client import DEFAULT_BASE_URL, API_URL_ENV  # noqa: E402
    check(DEFAULT_BASE_URL == "http://127.0.0.1:8000",
          "10.3 Default base URL is http://127.0.0.1:8000",
          f"got {DEFAULT_BASE_URL}")
    check(API_URL_ENV == "CODEPILOT_API_URL",
          "10.4 Env var name is CODEPILOT_API_URL", f"got {API_URL_ENV}")


def verify_backend_unavailable() -> None:
    """Part 8: connection failure is handled without a traceback."""
    section("11. backend unavailable (connection refused)")
    dead_port = free_port()
    dead_url = f"http://127.0.0.1:{dead_port}"
    code, out, err = run_cli(["health"], env_extra={"CODEPILOT_API_URL": dead_url})
    check(code == 1, "11.1 health against a closed port exits 1", f"exit={code}")
    check(f"{ERROR}" in err, "11.2 [ERROR] banner shown", err[:160])
    check("not reachable" in err, "11.3 Error explains the API is unreachable", err)
    check(dead_url in err, "11.4 Error names the configured URL",
          f"stderr: {err[:160]}")
    check("Traceback" not in err,
          "11.5 No Python traceback shown")


def verify_client_boundary_stub() -> None:
    """Client-boundary checks: exact payloads, top_k, malformed JSON.

    A stub HTTP server with a deterministic response lets us assert what the
    client ACTUALLY sends over HTTP (clone/index/ask request bodies and the
    top_k field) and how it reacts to a malformed JSON response -- all
    without depending on the embedding model or index state.
    """
    section("12. client boundary (real CLI -> required back-end behavior)")
    ok_payload = json.dumps({
        "answer": "auth appears in src/auth.py lines 10-15",
        "repository_name": "demo-repository",
        "question": "Where is user authentication handled?",
        "sources": [{
            "file_path": "src/auth.py", "language": "Python",
            "start_line": 10, "end_line": 15, "relevance_score": 0.87,
        }],
        "insufficient_context": False,
        "model_name": "mock-llm",
        "provider_name": "mock",
        "retrieved_count": 1,
        "context_included_count": 1,
    })
    url = stub_url()
    try:
        STUB_STATE["next_body"] = ok_payload
        code, out, err = run_cli(
            ["ask", "demo-repository", AUTH_QUESTION, "--top-k", "7"],
            env_extra={"CODEPILOT_API_URL": url},
        )
        path, payload = last_stub_request()
        check(path == "/rag/query", "12.1 ask calls POST /rag/query",
              f"got {path}")
        check(payload.get("repository_name") == "demo-repository",
              "12.2 ask body carries repository_name", f"{payload}")
        check(payload.get("question") == AUTH_QUESTION,
              "12.3 ask body carries the question")
        check(payload.get("top_k") == 7,
              "12.4 --top-k 7 is transmitted over HTTP",
              f"got {payload.get('top_k')!r}")
        check(code == 0 and "src/auth.py" in out,
              "12.5 ask renders the stub answer with the source path")

        STUB_STATE["next_body"] = ok_payload
        code, out, err = run_cli(
            ["ask", "demo-repository", AUTH_QUESTION],
            env_extra={"CODEPILOT_API_URL": url},
        )
        _, payload2 = last_stub_request()
        check("top_k" not in payload2,
              "12.6 top_k is omitted when --top-k not supplied", f"{payload2}")

        STUB_STATE["next_body"] = json.dumps({
            "success": True, "repository_name": "demo-repository",
            "local_path": "indexed_repos/demo-repository/source",
            "message": "Repository already exists",
        })
        code, out, err = run_cli(["clone", "https://github.com/x/demo-repository"],
                                 env_extra={"CODEPILOT_API_URL": url})
        path, payload = last_stub_request()
        check(path == "/repository/clone", "12.7 clone calls POST /repository/clone")
        check(payload.get("github_url") == "https://github.com/x/demo-repository",
              "12.8 clone body carries github_url", f"{payload}")

        STUB_STATE["next_body"] = json.dumps({
            "success": True, "repository_name": "demo-repository",
            "files_scanned": 4, "chunks_generated": 5,
            "embeddings_generated": 5, "vectors_stored": 5,
            "collection_count": 5, "repository_vectors": 5,
            "duration": 0.5, "embedding_model": "test-model",
            "message": "Successfully indexed repository",
        })
        code, out, err = run_cli(["index", "demo-repository"],
                                 env_extra={"CODEPILOT_API_URL": url})
        path, payload = last_stub_request()
        check(path == "/repository/index", "12.9 index calls POST /repository/index")
        check(payload.get("repository_name") == "demo-repository",
              "12.10 index body carries repository_name", f"{payload}")
        check("Vectors stored: 5" in out,
              "12.11 Client renders a stub index response", out)

        # 12.13 health against a stub with an unexpected payload shape.
        STUB_STATE["next_body"] = "[1,2,3]"
        code, out, err = run_cli(["health"], env_extra={"CODEPILOT_API_URL": url})
        check(code == 1 and f"{ERROR}" in err,
              "12.12 non-object JSON response is handled cleanly",
              f"exit={code} err={err[:120]}")

        # Malformed JSON response -> protocol error, no traceback.
        STUB_STATE["next_body"] = '{"answer": '
        code, out, err = run_cli(["ask", "demo-repository", "q"],
                                 env_extra={"CODEPILOT_API_URL": url})
        check(code == 1 and f"{ERROR}" in err,
              "12.13 malformed JSON response handled, exit 1",
              f"exit={code} err={err[:120]}")
        check("Traceback" not in err, "12.14 malformed JSON shows no traceback")
    finally:
        stub_stop()


def verify_no_artifacts(baseline: set) -> None:
    """Part 11: no secrets / generated artifacts were added to the repo."""
    section("13. no secrets / generated artifacts inside the repository")

    after = git_porcelain()
    unexpected = (after - baseline) - EXPECTED_NEW_PATHS
    check(not unexpected, "13.1 Only intended files are new/changed in git",
          f"unexpected: {sorted(unexpected)}")

    env_hits = [
        str(p.relative_to(REPO_ROOT))
        for p in REPO_ROOT.rglob(".env")
        if ".venv" not in str(p).lower()
    ]
    check(not env_hits, "13.2 No .env file exists in the project",
          f"found: {env_hits}")

    real_chroma = BACKEND_DIR / "chroma_db"
    check(not real_chroma.exists(), "13.3 No chroma_db created under backend/",
          f"found: {real_chroma}")

    repos_base = BACKEND_DIR / "indexed_repos"
    existing = set(os.listdir(repos_base)) if repos_base.exists() else set()
    sandbox_leaked = existing & {"demo-repository"}
    check(not sandbox_leaked,
          "13.4 No sandbox repositories created inside backend/indexed_repos",
          f"found: {sorted(sandbox_leaked)}")

    server_log = SERVER_LOG
    if server_log is not None and server_log.exists():
        text = server_log.read_text(encoding="utf-8", errors="replace")
        check("Traceback" not in text,
              "13.5 Backend server log contains no Python tracebacks",
              "traceback found in server.log")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all Sprint 10A CLI verifications."""
    baseline = git_porcelain()
    scratch = Path(tempfile.mkdtemp(prefix="codepilot_sprint10a_"))

    # Fixtures for the real-backend tests live in the server sandbox.
    write_repo(scratch, REPO_A, REPO_A_CONTENTS)

    start_backend(scratch)
    try:
        if not wait_backend_ready():
            log_text = SERVER_LOG.read_text(encoding="utf-8", errors="replace") \
                if SERVER_LOG.exists() else ""
            check(False, "Backend did not start for CLI verification",
                  log_text[-2000:])
            final_verdict()

        verify_health_command()
        verify_clone_command()
        verify_index_command()
        verify_index_errors()
        verify_ask_command()
        verify_insufficient_context()
        verify_nonexistent_repository_ask()
        verify_top_k()
        verify_validation_errors()
        verify_url_configuration()
        verify_backend_unavailable()
        verify_client_boundary_stub()
        verify_no_artifacts(baseline)
    finally:
        stop_backend()
        # The server is stopped, so the temp sandbox can be removed. Ignore
        # errors (e.g. a lingering file handle) -- it lives in the system
        # temp directory anyway and never inside the repo.
        shutil.rmtree(scratch, ignore_errors=True)

    final_verdict()


if __name__ == "__main__":
    main()