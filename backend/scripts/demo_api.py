"""End-to-end API verification for Sprint 9 (backend API integration).

This script verifies the ACTUAL FastAPI application by making HTTP/API calls
through FastAPI's TestClient rather than calling internal classes directly.

Verifications:
    A. GET /health returns a healthy status.
    B. Local test repositories are created (no network, no GitHub).
    C. A nonexistent repository returns HTTP 404 on /repository/index.
    D. Indexing the demo repository through POST /repository/index.
    E. Response reports repository name, files scanned, chunks generated,
       embeddings generated, vectors stored, and the repository vector count.
    F. Re-indexing does not create duplicate vector records.
    G. POST /rag/query with the auth question (answer, repo, question,
       sources with file paths and line ranges, provider/model metadata).
    H. Insufficient-context query -> no fabricated repository context.
    I. Repository isolation between two repositories.
    J. Stale-vector replacement on re-index: a file that shrinks below its
       old chunk count no longer leaves a stale chunk, a deleted file's
       vectors are removed, the repository vector count matches the current
       repository state, and querying never returns deleted-file chunks.
    K. RAG repository semantics: a nonexistent repository returns HTTP 404,
       an existing-but-unindexed repository returns HTTP 404, and an indexed
       repository with no relevant context returns HTTP 200 with
       insufficient_context=true (never conflated with a 404).
    L. Request validation (empty repository name, empty question, invalid
       top_k, malformed request bodies).
    M. No secrets / no generated artifacts created inside the repository.

The output uses ASCII-only status indicators ([PASS]/[FAIL]/[ERROR]) for
Windows cp1252 console compatibility. The script exits 0 only when every
verification passes.

Run from the backend directory:
    python scripts/demo_api.py
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

# Run everything inside a throwaway sandbox BEFORE importing the app so the
# container's relative paths (indexed_repos/, chroma_db/) resolve inside the
# sandbox and never touch real project data.
_SCRATCH = tempfile.mkdtemp(prefix="codepilot_sprint9_")
os.chdir(_SCRATCH)

# Isolate the vector store and repositories before importing the app so the
# container picks up the sandbox locations at construction time.
os.environ["CHROMA_PERSIST_DIR"] = os.path.join(_SCRATCH, "chroma")
os.environ["COLLECTION_NAME"] = "sprint9_api_demo"

# Ensure the mock provider is used (default) so no API key / network / cost.
os.environ.pop("LLM_PROVIDER", None)
os.environ.pop("LLM_API_KEY", None)

# Make app.* and main importable when run from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Suppress noisy INFO logs BEFORE importing the app: main.py calls
# logging.basicConfig(INFO) at import, and basicConfig is a no-op once the
# root logger already has handlers, so this must run first. Warnings (e.g.
# no LLM_API_KEY) are still shown.
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

from fastapi.testclient import TestClient  # noqa: E402
from main import app  # noqa: E402

PASS = "[PASS]"
FAIL = "[FAIL]"
ERROR = "[ERROR]"

client = TestClient(app)

# Repositories are created/looked-up under the same relative directory the
# container's RepositoryManager uses; after the chdir above this resolves to
# <scratch>/indexed_repos.
_repos_base = Path.cwd() / "indexed_repos"

_checks_run = 0
_checks_passed = 0


# ---------------------------------------------------------------------------
# Small helpers (imported lazily so the module header stays clean)
# ---------------------------------------------------------------------------

def json_dumps(obj) -> str:
    """Serialize an object to compact JSON."""
    import json
    return json.dumps(obj, sort_keys=True)


def ascii_only(text: str) -> bool:
    """Return True if text contains only ASCII characters."""
    return all(ord(c) < 128 for c in text)


# ---------------------------------------------------------------------------
# Verification helpers (ASCII-only output)
# ---------------------------------------------------------------------------

def check(condition: bool, label: str, detail: str = "") -> bool:
    """Record a verification outcome and print an ASCII-only verdict.

    Args:
        condition: True if the check passes.
        label: Short label of the check.
        detail: Optional detail printed on failure.

    Returns:
        True when the check passes, False otherwise.
    """
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
    """Print an ASCII-only section header."""
    print(f"\n{'-' * 66}")
    print(f" {title}")
    print("-" * 66)


def final_verdict() -> None:
    """Summarize results and set the process exit code (0 only if all pass)."""
    print(f"\n{'=' * 66}")
    print(f" {_checks_passed}/{_checks_run} checks passed (Sprint 9 API integration)")
    print("=" * 66)
    if _checks_run > 0 and _checks_passed == _checks_run:
        print(f"{PASS} ALL VERIFICATIONS PASSED (exit code 0)")
        sys.exit(0)
    print(f"{FAIL} SOME VERIFICATIONS FAILED (exit code 1)")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Fixture: small local repositories (no network)
# ---------------------------------------------------------------------------

REPO_A = "demo-repository"
REPO_B = "other-repository"

AUTH_QUESTION = "Where is user authentication handled?"

# Files unique to repository A (never appear in repository B).
REPO_A_FILES = {"src/db.py", "src/api/routes.py", "src/utils/helpers.py"}

# Files unique to repository B (never appear in repository A).
REPO_B_FILES = {"src/payments.py"}

REPO_A_CONTENTS = {
    "src/auth.py": (
        'def authenticate_user(username, password):\n'
        '    """Validate a username/password pair."""\n'
        "    user = users_db.get(username)\n"
        '    if user and user.check_password(password):\n'
        "        return user\n"
        "    return None\n"
    ),
    "src/db.py": (
        'def connect_database():\n'
        '    """Open the application database connection."""\n'
        "    import sqlite3\n"
        '    conn = sqlite3.connect("app.db")\n'
        "    return conn\n"
    ),
    "src/api/routes.py": (
        'def login_route(request):\n'
        '    """Handle POST /login."""\n'
        "    user = authenticate_user(request.username, request.password)\n"
        "    if user is None:\n"
        '        return error("invalid credentials")\n'
        "    return ok(session_token_for(user))\n"
    ),
    "src/utils/helpers.py": (
        'def tokenize(text):\n'
        '    """Split text into tokens."""\n'
        "    return text.split()\n"
    ),
}

REPO_B_CONTENTS = {
    "src/auth.py": (
        "def authenticate_user(username, password):\n"
        '    """Validate credentials against the Mongo store."""\n'
        "    user = user_table.find_one({'username': username})\n"
        "    return user is not None and bcrypt.checkpw(\n"
        "        password.encode(), user['hash'])\n"
    ),
    "src/payments.py": (
        "def process_refund(transaction_id, amount):\n"
        '    """Refund a transaction by identifier."""\n'
        "    tx = find_transaction(transaction_id)\n"
        "    if tx is None:\n"
        "        return False\n"
        "    return refund_gateway.refund(tx.token, amount)\n"
    ),
}


# Part J fixtures: a repository that is modified between index runs so stale
# vectors (shrunk files, deleted files) can be verified to disappear.
SHRINK_REPO = "shrink-repository"

SHRINK_CONTENTS_MANY = {
    "src/helpers.py": (
        'def helper_one(arg):\n'
        '    """Helper number one. Returns its argument unchanged."""\n'
        "    return arg\n"
        "\n"
        "def helper_two(arg):\n"
        '    """Helper number two. Doubles the argument."""\n'
        "    return arg * 2\n"
        "\n"
        "def helper_three(arg):\n"
        '    """Helper number three. Triples the argument."""\n'
        "    return arg * 3\n"
    ),
}

# The same file, but THREE functions collapsed into ONE: fewer chunks, same
# relative path, so the old chunk-2 ID (helpers.py) is no longer produced.
SHRINK_CONTENTS_ONE = {
    "src/helpers.py": (
        'def helper_single(arg):\n'
        '    """Single helper after the file shrank. Returns arg."""\n'
        "    return arg\n"
    ),
}

# A file that only exists in the FIRST state and is deleted before re-index.
SHRINK_DELETED_AFTER = {
    "src/doomed.py": (
        'def doomed_function():\n'
        '    """File that will be deleted before the second re-index."""\n'
        "    return \"doomed\"\n"
    ),
}

MAX_CHUNKS_OF_SHRINK = 3
DOOMED_FILE = "src/doomed.py"


def write_repo(repo_name: str, files: dict) -> None:
    """Write a small local repository and initialize git in it.

    Args:
        repo_name: Repository name.
        files: Mapping of relative path -> file content.
    """
    from git import Repo

    source = _repos_base / repo_name / "source"
    source.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        path = source / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    # RepositoryManager treats a directory as a clone only when it has a
    # .git directory; a local init is enough (no network involved).
    Repo.init(str(source))


def setup_repositories() -> None:
    """Create the two local repositories (Parts B and I)."""
    section("B. Create local test repositories (no GitHub / no network)")
    write_repo(REPO_A, REPO_A_CONTENTS)
    write_repo(REPO_B, REPO_B_CONTENTS)

    a_source = _repos_base / REPO_A / "source"
    b_source = _repos_base / REPO_B / "source"
    check(a_source.is_dir() and (a_source / ".git").exists(),
          "B1. Repository A exists as a cloned directory")
    check(b_source.is_dir() and (b_source / ".git").exists(),
          "B2. Repository B exists as a cloned directory")
    check(all((a_source / p).is_file() for p in ["src/auth.py", "src/db.py"]),
          "B3. Repository A contains the auth fixture files")


# ---------------------------------------------------------------------------
# Part A: health
# ---------------------------------------------------------------------------

def verify_health() -> None:
    """Part A: GET /health."""
    section("A. Health endpoint")
    try:
        response = client.get("/health")
    except Exception as e:  # noqa: BLE001
        check(False, "A1. GET /health succeeds", f"{type(e).__name__}: {e}")
        return
    body = response.json()
    check(response.status_code == 200, "A1. GET /health returns HTTP 200",
          f"got {response.status_code}")
    check(body.get("status") == "healthy", "A2. Status is 'healthy'",
          f"got {body.get('status')!r}")
    check(bool(body.get("timestamp")), "A3. Response includes a timestamp")
    check(ascii_only(json_dumps(body)), "A4. Health response is ASCII-only")


# ---------------------------------------------------------------------------
# Parts C-F: indexing behavior
# ---------------------------------------------------------------------------

def verify_nonexistent_index() -> None:
    """Part C: a repository that is not cloned locally returns an error."""
    section("C. Nonexistent repository /repository/index")
    try:
        response = client.post(
            "/repository/index",
            json={"repository_name": "does-not-exist"},
        )
    except Exception as e:  # noqa: BLE001
        check(False, "C1. /index for missing repo raises no exception",
              f"{type(e).__name__}: {e}")
        return
    check(response.status_code == 404,
          "C1. Missing repository returns HTTP 404",
          f"got {response.status_code}")
    detail = response.json().get("detail", "")
    check(bool(detail) and "not present" in detail,
          "C2. Error message says the repository is not present", detail)


def verify_index_repo_a() -> dict:
    """Parts D/E: index repository A and check the response fields.

    Returns:
        The index response body as a dict ({} on failure).
    """
    section(f"D. POST /repository/index for {REPO_A}")
    try:
        response = client.post(
            "/repository/index",
            json={"repository_name": REPO_A},
        )
    except Exception as e:  # noqa: BLE001
        check(False, "D1. POST /repository/index succeeds",
              f"{type(e).__name__}: {e}")
        return {}

    body = response.json()
    check(response.status_code == 200, "D1. Index returns HTTP 200",
          f"got {response.status_code}")
    if response.status_code != 200:
        print(f"     body: {json_dumps(body)}")
        return {}

    section("E. Index response reports structured statistics")
    check(body.get("success") is True,
          "E1. success is true", f"got {body.get('success')!r}")
    check(body.get("repository_name") == REPO_A,
          "E2. repository_name is preserved",
          f"got {body.get('repository_name')!r}")
    check(isinstance(body.get("files_scanned"), int)
          and body["files_scanned"] == 4,
          "E3. All 4 source files were scanned",
          f"got {body.get('files_scanned')!r}")
    check(isinstance(body.get("chunks_generated"), int)
          and body["chunks_generated"] == body.get("embeddings_generated"),
          "E4. chunks_generated equals embeddings_generated, both positive",
          f"chunks={body.get('chunks_generated')!r} "
          f"embeddings={body.get('embeddings_generated')!r}")
    check(isinstance(body.get("embeddings_generated"), int)
          and body["embeddings_generated"] > 0,
          "E5. embeddings_generated reported",
          f"got {body.get('embeddings_generated')!r}")
    check(isinstance(body.get("vectors_stored"), int)
          and body["vectors_stored"] > 0
          and body["vectors_stored"] == body["embeddings_generated"],
          "E6. vectors_stored reported and matches embeddings",
          f"got {body.get('vectors_stored')!r}")
    check(bool(body.get("message")), "E7. message is present")
    check(ascii_only(json_dumps(body)), "E8. Index response is ASCII-only")
    return body


def verify_reindex(body_a: dict) -> None:
    """Part F: re-indexing must not create duplicate vector records.

    Args:
        body_a: The response from the first index of repository A.
    """
    section(f"F. Re-index {REPO_A} (upsert, no duplicates)")
    first_count = body_a.get("collection_count", 0)
    first_stored = body_a.get("vectors_stored", 0)

    try:
        response = client.post(
            "/repository/index",
            json={"repository_name": REPO_A},
        )
    except Exception as e:  # noqa: BLE001
        check(False, "F1. Re-index raises no exception",
              f"{type(e).__name__}: {e}")
        return

    body = response.json()
    check(response.status_code == 200, "F1. Re-index returns HTTP 200",
          f"got {response.status_code}")
    check(body.get("success") is True and body.get("chunks_generated", 0) > 0,
          "F2. Re-index reports generated chunks again",
          json_dumps(body))
    check(body.get("collection_count") == first_count,
          "F3. Collection count unchanged (no duplicate vectors)",
          f"before={first_count} after={body.get('collection_count')}")
    check(body.get("vectors_stored") == first_stored,
          "F4. vectors_stored unchanged on re-index",
          f"before={first_stored} after={body.get('vectors_stored')}")


# ---------------------------------------------------------------------------
# Part G: RAG query over repository A
# ---------------------------------------------------------------------------

def verify_rag_query_a() -> None:
    """Part G: query repository A about authentication."""
    section(f"G. POST /rag/query for {REPO_A} ({AUTH_QUESTION!r})")
    try:
        response = client.post(
            "/rag/query",
            json={"repository_name": REPO_A, "question": AUTH_QUESTION},
        )
    except Exception as e:  # noqa: BLE001
        check(False, "G1. POST /rag/query raises no exception",
              f"{type(e).__name__}: {e}")
        return

    body = response.json()
    check(response.status_code == 200, "G1. Query returns HTTP 200",
          f"got {response.status_code}")
    if response.status_code != 200:
        print(f"     body: {json_dumps(body)}")
        return

    check(bool(body.get("answer", "").strip()), "G2. Answer is returned",
          f"got {body.get('answer', '')[:80]!r}")
    check(body.get("repository_name") == REPO_A,
          "G3. repository_name is preserved",
          f"got {body.get('repository_name')!r}")
    check(body.get("question") == AUTH_QUESTION,
          "G4. question is preserved",
          f"got {body.get('question')!r}")
    check(isinstance(body.get("sources"), list) and len(body["sources"]) > 0,
          "G5. Retrieved sources exist",
          f"got {len(body.get('sources', []))}")
    check(body.get("insufficient_context") is False,
          "G6. Auth question has sufficient context",
          f"got {body.get('insufficient_context')!r}")

    sources_ok = all(
        s.get("file_path") and "src/" in s.get("file_path", "")
        and isinstance(s.get("start_line"), int) and s["start_line"] >= 1
        and isinstance(s.get("end_line"), int)
        and s["end_line"] >= s["start_line"]
        and 0.0 <= s.get("relevance_score", -1.0) <= 1.0
        for s in body.get("sources", [])
    )
    check(sources_ok,
          "G7. Sources have valid file paths, line ranges, and relevance",
          json_dumps(body.get("sources", [])))

    auth_source = next(
        (s for s in body.get("sources", [])
         if "auth" in s.get("file_path", "").lower()),
        None,
    )
    check(auth_source is not None,
          "G8. A source references the authentication code",
          json_dumps(body.get("sources", [])))

    check(bool(body.get("provider_name")) and bool(body.get("model_name")),
          "G9. Provider/model information is exposed",
          f"provider={body.get('provider_name')!r} model={body.get('model_name')!r}")
    check(body.get("provider_name") == "mock"
          and body.get("model_name") == "mock-llm",
          "G10. Zero-cost mock provider/model recorded",
          f"provider={body.get('provider_name')!r} model={body.get('model_name')!r}")

    serialized = json_dumps(body).lower()
    check("embedding" not in serialized and "chroma" not in serialized,
          "G11. No raw embedding vectors or ChromaDB objects exposed")
    check(ascii_only(serialized), "G12. Query response is ASCII-only")


# ---------------------------------------------------------------------------
# Part H: insufficient context
# ---------------------------------------------------------------------------

def verify_insufficient_context() -> None:
    """Part H: out-of-context questions must not fabricate an answer."""
    section("H. Insufficient-context question")
    question = "Where is blockchain mining implemented?"
    try:
        response = client.post(
            "/rag/query",
            json={"repository_name": REPO_A, "question": question},
        )
    except Exception as e:  # noqa: BLE001
        check(False, "H1. Out-of-context query raises no exception",
              f"{type(e).__name__}: {e}")
        return

    body = response.json()
    check(response.status_code == 200, "H1. Out-of-context query returns 200",
          f"got {response.status_code}")
    check("blockchain" not in json_dumps(body.get("sources", [])).lower(),
          "H2. No fabricated blockchain source is retrieved",
          json_dumps(body.get("sources", [])))
    check(body.get("insufficient_context") is True,
          "H3. Response flagged insufficient_context",
          f"got {body.get('insufficient_context')!r}")
    answer_lower = body.get("answer", "").lower()
    check("insufficient" in answer_lower or "no relevant" in answer_lower,
          "H4. Answer explicitly states insufficient evidence",
          body.get("answer", "")[:120])
    check(ascii_only(json_dumps(body)), "H5. Insufficient-context response is ASCII-only")


# ---------------------------------------------------------------------------
# Part K: RAG repository semantics (404 vs insufficient_context)
# ---------------------------------------------------------------------------

def verify_rag_repository_semantics() -> None:
    """Part K: RAG distinguishes repository-not-available from no-context.

    A nonexistent repository -> 404. An existing-but-unindexed repository ->
    404. An indexed repository with no relevant context -> 200 +
    insufficient_context=true.
    """
    section("K. RAG repository semantics (404 vs insufficient context)")

    # Create SHRINK_REPO locally WITHOUT indexing it, so part B below has a
    # repository that "exists locally but has not been indexed".
    write_repo(SHRINK_REPO, SHRINK_CONTENTS_ONE)

    # A) Repository does not exist locally -> HTTP 404.
    try:
        response = client.post(
            "/rag/query",
            json={
                "repository_name": "ghost-repository",
                "question": AUTH_QUESTION,
            },
        )
    except Exception as e:  # noqa: BLE001
        check(False, "K1. Nonexistent repo query raises no exception",
              f"{type(e).__name__}: {e}")
        return
    check(response.status_code == 404,
          "K1. Nonexistent repository returns HTTP 404",
          f"got {response.status_code}")
    detail = response.json().get("detail", "")
    check("not found" in detail.lower() or "has not been indexed" in detail.lower(),
          "K2. 404 message explains the repository is unavailable", detail)

    # B) Repository exists locally but has not been indexed -> HTTP 404.
    try:
        response = client.post(
            "/rag/query",
            json={
                "repository_name": SHRINK_REPO,
                "question": AUTH_QUESTION,
            },
        )
    except Exception as e:  # noqa: BLE001
        check(False, "K3. Unindexed repo query raises no exception",
              f"{type(e).__name__}: {e}")
        return
    check(response.status_code == 404,
          "K3. Existing-but-unindexed repository returns HTTP 404",
          f"got {response.status_code} (body: {response.text[:120]})")

    # C) Indexed repository with a no-relevant-context question must NOT be
    #    conflated with a 404: it is HTTP 200 + insufficient_context=true.
    try:
        client.post(
            "/repository/index",
            json={"repository_name": SHRINK_REPO},
        )
        response = client.post(
            "/rag/query",
            json={
                "repository_name": SHRINK_REPO,
                "question": "Where is the blockchain mining loop defined?",
            },
        )
    except Exception as e:  # noqa: BLE001
        check(False, "K4. Indexed no-context query raises no exception",
              f"{type(e).__name__}: {e}")
        return

    body = response.json()
    check(response.status_code == 200,
          "K4. Indexed no-context query returns HTTP 200 (not 404)",
          f"got {response.status_code}")
    check(body.get("insufficient_context") is True,
          "K5. insufficient_context=true for the indexed repo",
          f"got {body.get('insufficient_context')!r}")


# ---------------------------------------------------------------------------
# Part I: repository isolation
# ---------------------------------------------------------------------------

def verify_repository_isolation() -> None:
    """Part I: answers for repo A must not bleed into repo B and vice versa.

    The mock provider echoes only the file paths and line ranges from the
    prompt (never full code content), so isolation is verified by ensuring
    a repository's unique file paths never appear in the other's answer.
    """
    section(f"I. Repository isolation ({REPO_A} vs {REPO_B})")

    # Index repository B so it becomes queryable.
    try:
        response = client.post(
            "/repository/index",
            json={"repository_name": REPO_B},
        )
    except Exception as e:  # noqa: BLE001
        check(False, "I1. Index repository B raises no exception",
              f"{type(e).__name__}: {e}")
        return
    check(response.status_code == 200, "I1. Index repository B succeeds",
          f"got {response.status_code} "
          f"{response.json().get('message', '' if response.status_code == 200 else response.text[:80])}")

    # Ask the auth question in repository B.
    try:
        response_b = client.post(
            "/rag/query",
            json={"repository_name": REPO_B, "question": AUTH_QUESTION},
        )
        response_a = client.post(
            "/rag/query",
            json={"repository_name": REPO_A, "question": AUTH_QUESTION},
        )
    except Exception as e:  # noqa: BLE001
        check(False, "I2. Queries raise no exception",
              f"{type(e).__name__}: {e}")
        return

    body_b = response_b.json()
    body_a = response_a.json()

    check(response_b.status_code == 200 and response_a.status_code == 200,
          "I2. Both queries return HTTP 200",
          f"B={response_b.status_code} A={response_a.status_code}")

    answer_a = body_a.get("answer", "")
    answer_b = body_b.get("answer", "")

    check(body_b.get("repository_name") == REPO_B,
          "I3. Repo B response records repository B",
          f"got {body_b.get('repository_name')!r}")
    check(body_a.get("repository_name") == REPO_A,
          "I4. Repo A response records repository A",
          f"got {body_a.get('repository_name')!r}")

    leaked_into_b = any(f in answer_b for f in REPO_A_FILES)
    check(not leaked_into_b,
          "I5. Repo A's unique files do not leak into repo B's answer",
          answer_b[:200])
    leaked_into_a = any(f in answer_a for f in REPO_B_FILES)
    check(not leaked_into_a,
          "I6. Repo B's unique files do not leak into repo A's answer",
          answer_a[:200])

    # Affirmative citation check, tolerant of an insufficient-context answer
    # (which by definition cites no file). If sources WERE returned, they must
    # point at the repository's own (or shared) files, never the other repo's.
    b_cites_own = (
        body_b.get("insufficient_context") is True
        or any(f in answer_b for f in REPO_B_FILES)
        or "src/auth.py" in answer_b
    )
    check(b_cites_own,
          "I7. Repo B's answer cites repo B context (or flags insufficient)",
          answer_b[:200])
    a_cites_own = (
        body_a.get("insufficient_context") is True
        or any(f in answer_a for f in REPO_A_FILES)
        or "src/auth.py" in answer_a
    )
    check(a_cites_own,
          "I8. Repo A's answer cites repo A context (or flags insufficient)",
          answer_a[:200])


# ---------------------------------------------------------------------------
# Part J: stale-vector replacement on re-index
# ---------------------------------------------------------------------------

def index_repo_json(repo_name: str) -> dict:
    """POST /repository/index and return the JSON body.

    Args:
        repo_name: Repository to index.

    Returns:
        Response body as a dict, or {} on any failure.
    """
    try:
        response = client.post(
            "/repository/index",
            json={"repository_name": repo_name},
        )
    except Exception as e:  # noqa: BLE001
        check(False, f"Index {repo_name} raises no exception",
              f"{type(e).__name__}: {e}")
        return {}
    if response.status_code != 200:
        check(False, f"Index {repo_name} returns HTTP 200",
              f"got {response.status_code}: {response.text[:120]}")
        return {}
    return response.json()


def verify_stale_vector_replacement() -> None:
    """Part J: re-index replaces the repository's stale vectors.

    Flow:
        J1. Index a repository whose helpers.py yields MAX_CHUNKS_OF_SHRINK
            chunks (plus a doomed file) and record the repository vector
            count.
        J2. Confirm the helper-function chunks are queryable.
        J3. Shrink helpers.py to a single function (fewer chunks) and
            re-index. The old chunk(s) must be gone and the repository
            vector count must match the new chunk count.
        J4. Delete a file, re-index again. Its vectors must be gone, the
            repository vector count must match the current repository state,
            and querying must never return deleted-file chunks.
    """
    section(f"J. Stale-vector replacement on re-index ({SHRINK_REPO})")

    # First index: helpers.py yields MAX_CHUNKS_OF_SHRINK chunks + doomed.py.
    write_repo(
        SHRINK_REPO,
        {**SHRINK_CONTENTS_MANY, **SHRINK_DELETED_AFTER},
    )
    body1 = index_repo_json(SHRINK_REPO)
    vars_count_1 = body1.get("repository_vectors", 0)
    chunks_1 = body1.get("chunks_generated", 0)
    check(body1.get("success") is True
          and chunks_1 == MAX_CHUNKS_OF_SHRINK + 1,
          "J1. First index stores helpers (3 chunks) + doomed.py (1 chunk)",
          json_dumps({"chunks_generated": chunks_1,
                      "repository_vectors": vars_count_1}))
    check(vars_count_1 == chunks_1,
          "J1. repository_vectors matches chunks after the first index",
          f"vectors={vars_count_1} chunks={chunks_1}")

    # The helper chunks are queryable before the file shrinks.
    try:
        response = client.post(
            "/rag/query",
            json={
                "repository_name": SHRINK_REPO,
                "question": "How are the helper functions implemented?",
            },
        )
    except Exception as e:  # noqa: BLE001
        check(False, "J2. Query over the 3-helper file raises no exception",
              f"{type(e).__name__}: {e}")
        return
    body = response.json()
    helper_paths = [s.get("file_path") for s in body.get("sources", [])]
    check(
        response.status_code == 200
        and any("helpers.py" in (p or "") for p in helper_paths),
        "J2. Helper chunks are retrievable before the file shrinks",
        f"status={response.status_code} sources={helper_paths}",
    )

    # Shrink helpers.py from 3 functions to 1, then re-index. The old helper
    # chunk(s) must no longer exist, and the repo must have just 2 vectors
    # (1 helper + 1 doomed) matching the current repository state.
    write_repo(SHRINK_REPO, SHRINK_CONTENTS_ONE)
    body2 = index_repo_json(SHRINK_REPO)
    vars_count_2 = body2.get("repository_vectors", 0)
    chunks_2 = body2.get("chunks_generated", 0)
    check(body2.get("success") is True and chunks_2 == 2,
          "J3. Shrunk file re-indexes with fewer chunks (1 helper + 1 doomed)",
          f"chunks_generated={chunks_2}")
    check(vars_count_2 == chunks_2 == 2,
          "J3. No stale helper chunk remains (repository_vectors == 2)",
          f"vectors={vars_count_2} chunks={chunks_2}")

    # Delete doomed.py and re-index. Its vectors must be gone; the repository
    # must now hold exactly 1 vector, matching the single-function file.
    write_repo(SHRINK_REPO, SHRINK_CONTENTS_ONE)
    (Path.cwd() / "indexed_repos" / SHRINK_REPO / "source" / DOOMED_FILE) \
        .unlink()

    body3 = index_repo_json(SHRINK_REPO)
    vars_count_3 = body3.get("repository_vectors", 0)
    chunks_3 = body3.get("chunks_generated", 0)
    check(body3.get("success") is True and chunks_3 == 1,
          "J4. Re-index after deleting a file generates 1 chunk",
          f"chunks_generated={chunks_3}")
    check(vars_count_3 == 1,
          "J4. Deleted-file vectors are removed (repository_vectors == 1)",
          f"vectors={vars_count_3}")

    # Querying must never return deleted-file chunks.
    try:
        response = client.post(
            "/rag/query",
            json={
                "repository_name": SHRINK_REPO,
                "question": AUTH_QUESTION,
            },
        )
    except Exception as e:  # noqa: BLE001
        check(False, "J5. Query after deletion raises no exception",
              f"{type(e).__name__}: {e}")
        return
    body = response.json()
    sources = body.get("sources", [])
    deleted_present = any(
        "doomed" in (s.get("file_path") or "").lower() for s in sources
    )
    check(response.status_code == 200 and not deleted_present,
          "J5. Querying never returns deleted-file chunks",
          f"status={response.status_code} sources="
          f"{[s.get('file_path') for s in sources]}")


# ---------------------------------------------------------------------------
# Part L: request validation
# ---------------------------------------------------------------------------

def verify_validation() -> None:
    """Part L: invalid request bodies are rejected by the API layer."""
    section("L. Request validation")

    cases = [
        ("L1. Empty repository name (index) -> 422",
         "/repository/index", {"repository_name": ""}),
        ("L2. Whitespace repository name (index) -> 422",
         "/repository/index", {"repository_name": "   "}),
        ("L4. Empty question (rag) -> 422",
         "/rag/query", {"repository_name": REPO_A, "question": ""}),
        ("L5. Whitespace question (rag) -> 422",
         "/rag/query", {"repository_name": REPO_A, "question": "   \t "}),
        ("L6. invalid top_k 0 (rag) -> 422",
         "/rag/query",
         {"repository_name": REPO_A, "question": AUTH_QUESTION, "top_k": 0}),
        ("L7. invalid top_k -3 (rag) -> 422",
         "/rag/query",
         {"repository_name": REPO_A, "question": AUTH_QUESTION, "top_k": -3}),
        ("L8. missing required fields (rag) -> 422",
         "/rag/query", {"unexpected": True}),
    ]
    for label, url, payload in cases:
        response = client.post(url, json=payload)
        check(response.status_code == 422, label, f"got {response.status_code}")

    # Malformed body (not JSON at all).
    response = client.post(
        "/rag/query",
        content="not-json",
        headers={"content-type": "application/json"},
    )
    check(response.status_code == 422,
          "L9. Malformed JSON body (rag) -> 422",
          f"got {response.status_code}")


# ---------------------------------------------------------------------------
# Part K: no secrets / no generated artifacts inside the repo
# ---------------------------------------------------------------------------

def verify_no_generated_artifacts() -> None:
    """Part M: the run must not create artifacts inside the project.

    The demo sandbox lives in the system temp directory and the chroma
    store is redirected to it, so nothing under backend/ should have been
    created by this run.
    """
    section("M. No secrets / no generated artifacts inside the repo")
    backend_dir = Path(__file__).resolve().parent.parent

    # The real backend must not contain sandbox repositories.
    backend_repos = backend_dir / "indexed_repos"
    created_here = [
        p for p in [REPO_A, REPO_B]
        if (backend_repos / p).exists()
    ]
    check(not created_here,
          "M1. No sandbox repository directories inside backend/",
          f"found: {created_here}")

    # The real backend must not contain a chroma_db created by this run.
    real_chroma = backend_dir / "chroma_db"
    check(not real_chroma.exists(),
          "M2. No chroma_db directory inside backend/",
          f"found: {real_chroma}")

    # .env.example (the only allowed config placeholder) is still present.
    env_example = backend_dir / ".env.example"
    check(env_example.exists(),
          "M3. .env.example still present",
          f"missing: {env_example}")

    # No .env / keys created anywhere inside the repo.
    env_hits = [
        str(p.relative_to(Path(__file__).resolve().parent.parent.parent))
        for p in Path(__file__).resolve().parent.parent.parent.rglob(".env")
        if ".venv" not in str(p) and "node_modules" not in str(p)
    ]
    check(not env_hits, "M4. No .env file inside the project",
          f"found: {env_hits}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all Sprint 9 verifications and exit with the verdict."""
    verify_health()
    setup_repositories()
    verify_nonexistent_index()
    body_a = verify_index_repo_a()
    if body_a:
        verify_reindex(body_a)
    verify_rag_query_a()
    verify_insufficient_context()
    verify_rag_repository_semantics()
    verify_repository_isolation()
    verify_stale_vector_replacement()
    verify_validation()
    verify_no_generated_artifacts()
    final_verdict()


if __name__ == "__main__":
    main()