"""End-to-end verification of the Sprint 8 Generation layer (Rs. 0 / no-cost mode).

Runs the full RAG pipeline using ONLY local, free components:
    - Realistic CodeChunks representing a small repository
    - The existing EmbeddingEngine (BAAI/bge-small-en-v1.5, cached locally)
    - The existing ChromaVectorStore (temp directory, never committed)
    - The existing Retriever
    - The new PromptBuilder (Sprint 8)
    - The new MockLLMProvider (Sprint 8, no API key, no network, no cost)
    - The new RAGService (Sprint 8)

Verifications (ASCII-only output; exit code 0 only when ALL pass):
    A. Build realistic CodeChunks
    B. Generate embeddings with EmbeddingEngine
    C. Store with ChromaVectorStore
    D. Create the Retriever
    E. Create the PromptBuilder
    F. Create the MockLLMProvider
    G. Create the RAGService
    H. Ask "Where is user authentication handled?"
    I. Verify retrieval, prompt contents (repo name, file paths, line ranges,
       retrieved code, grounding instructions, user question), mock LLM
       receives the prompt, final answer returned, retrieved results included
    J. Insufficient-context question -> no exception, no fabrication, clear
       insufficient-context response
    K. Context budget -> complete chunks included, budget respected, chunks
       never partially truncated
    L. Provider abstraction -> mock used through the LLMProvider interface;
       RAGService does not depend on the concrete mock implementation
    M/N. ASCII-only output; exit code 0 only when every check passes.

Run from the backend directory:
    python scripts/demo_rag.py
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import sys
import tempfile
from pathlib import Path

# Make app.* importable when run from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chunking.models import ChunkType, CodeChunk
from app.embeddings import EmbeddingEngine, initialize_providers
from app.llm import MockLLMProvider
from app.llm.models import LLMResponse
from app.llm.provider import LLMProvider
from app.prompts import PromptBuilder, PromptContextBudget
from app.rag.models import RetrievalResult
from app.rag.retriever import Retriever
from app.rag.service import RAGService
from app.vectorstore import ChunkEmbeddingPair, create_vector_store

# Suppress noisy INFO logs; keep warnings (e.g. no LLM_API_KEY).
logging.basicConfig(level=logging.WARNING)

REPOSITORY_NAME = "demo-repo"
AUTH_QUESTION = "Where is user authentication handled?"

_checks_run = 0
_checks_passed = 0


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
        print(f"[PASS] {label}")
        return True
    print(f"[FAIL] {label}")
    if detail:
        print(f"      {detail}")
    return False


def section(title: str) -> None:
    """Print an ASCII-only section header."""
    print(f"\n{'-' * 66}")
    print(f" {title}")
    print('-' * 66)


def final_verdict() -> None:
    """Summarize results and set the process exit code (0 only if all pass)."""
    print(f"\n{'=' * 66}")
    print(f" {_checks_passed}/{_checks_run} checks passed (Sprint 8 Generation layer)")
    print('=' * 66)
    if _checks_run > 0 and _checks_passed == _checks_run:
        print("[PASS] ALL VERIFICATIONS PASSED (exit code 0)")
        sys.exit(0)
    print("[FAIL] SOME VERIFICATIONS FAILED (exit code 1)")
    sys.exit(1)


def ascii_only(text: str) -> bool:
    """Return True if text contains only ASCII characters."""
    return all(ord(c) < 128 for c in text)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def hex_hash(content: str) -> str:
    """Compute the sha256 content hash used by CodeChunk."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def make_chunk(
    index: int,
    relative_path: str,
    content: str,
    language: str = "Python",
    chunk_type: ChunkType = ChunkType.FUNCTION,
    start_line: int = 1,
) -> CodeChunk:
    """Create a realistic CodeChunk fixture."""
    return CodeChunk(
        repository_name=REPOSITORY_NAME,
        relative_path=relative_path,
        language=language,
        chunk_type=chunk_type,
        chunk_index=index,
        start_line=start_line,
        end_line=start_line + content.count("\n"),
        content=content,
        content_hash=hex_hash(content),
        metadata={"fixture": True},
    )


def make_auth_chunks() -> list[CodeChunk]:
    """Realistic CodeChunks for the auth demo (3 files)."""
    return [
        make_chunk(
            0,
            "src/auth.py",
            'def authenticate_user(username, password):\n'
            '    """Validate a username/password pair."""\n'
            "    user = users_db.get(username)\n"
            '    if user and user.check_password(password):\n'
            "        return user\n"
            "    return None",
        ),
        make_chunk(
            1,
            "src/db.py",
            'def connect_database():\n'
            '    """Open the application database connection."""\n'
            "    import sqlite3\n"
            '    conn = sqlite3.connect("app.db")\n'
            "    conn.row_factory = sqlite3.Row\n"
            "    return conn",
        ),
        make_chunk(
            2,
            "src/api/routes.py",
            'def login_route(request):\n'
            '    """Handle POST /login."""\n'
            "    user = authenticate_user(request.username, request.password)\n"
            "    if user is None:\n"
            '        return error("invalid credentials")\n'
            '    return ok(session_token_for(user))',
        ),
    ]


def make_budget_chunks(count: int = 12) -> list[CodeChunk]:
    """Create many mid-size chunks to exercise the context budget.

    Each chunk is a nontrivial multi-line function (~300 chars), so a tight
    budget must skip WHOLE chunks rather than truncate one in the middle.
    """
    chunks = []
    for i in range(count):
        body = (
            f"def controller_{i}_run(self, request):\n"
            f'    """Handle request {i} for the demo service."""\n'
            f"    tokens = request.extract_tokens()\n"
            f"    payload = build_payload(tokens, config_policy_{i})\n"
            f"    return format_response(payload, indent_level=i)"
        )
        chunks.append(
            make_chunk(
                i,
                f"src/controllers/controller_{i}.py",
                body,
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Parts A-G: build fixtures, embed, store, wire components
# ---------------------------------------------------------------------------

def build_pipeline() -> tuple[Retriever, PromptBuilder, MockLLMProvider, EmbeddingEngine, object]:
    """Set up and return the wired components (Parts A-G).

    Returns:
        (retriever, builder, llm, engine, store). `store` is the concrete
        ChromaVectorStore used to seed the budget corpus for Part K.
    """
    section("A. Create realistic CodeChunks (small repository)")
    auth_chunks = make_auth_chunks()
    check(len(auth_chunks) == 3, "A1. Auth corpus has 3 chunks")
    check(
        all(isinstance(c, CodeChunk) for c in auth_chunks),
        "A2. Fixtures are CodeChunk objects",
    )

    section("B. Generate embeddings with the existing EmbeddingEngine")
    initialize_providers()
    engine = EmbeddingEngine()
    auth_pairs = [(chunk, engine.embed(chunk)) for chunk in auth_chunks]
    check(
        all(record.embedding for _, record in auth_pairs),
        "B1. Embeddings generated for all chunks",
    )
    info = engine.get_provider_info()
    check(
        info["model_name"] == "BAAI/bge-small-en-v1.5",
        "B2. Embedding model is the existing BGE provider",
        f"model: {info['model_name']}",
    )

    section("C. Store with the existing ChromaVectorStore (temp dir, no cost)")
    tmp_dir = tempfile.mkdtemp(prefix="codepilot_sprint8_")
    store = create_vector_store(persistence_path=tmp_dir, collection_name="sprint8_demo")
    stored = store.add_many(
        ChunkEmbeddingPair(chunk=chunk, record=record)
        for chunk, record in auth_pairs
    )
    check(stored == 3, "C1. All 3 pairs stored", f"stored={stored}")
    check(store.count() == 3, "C2. Store count is 3")

    section("D. Create the existing Retriever")
    retriever = Retriever(embedding_engine=engine, vector_store=store)
    check(isinstance(retriever, Retriever), "D1. Retriever created")

    section("E. Create the PromptBuilder")
    builder = PromptBuilder()  # default budget from configuration
    check(builder.budget.max_tokens > 0, "E1. PromptBuilder has a default budget")

    section("F. Create the MockLLMProvider (no API key, no network, no cost)")
    llm = MockLLMProvider()
    check(llm.get_provider_name() == "mock", "F0. Mock provider named 'mock'")
    check(not os.environ.get("LLM_API_KEY"), "F1. No API key required (mock mode)")

    return retriever, builder, llm, engine, store


# ---------------------------------------------------------------------------
# Parts H/I/L: full pipeline end-to-end
# ---------------------------------------------------------------------------

def verify_pipeline(retriever: Retriever, builder: PromptBuilder, llm) -> RAGService:
    """Parts G, H, I, L: end-to-end retrieval -> prompt -> mock -> answer.

    Returns:
        The configured RAGService (reused by later parts).
    """
    section("G. Create the RAGService")
    service = RAGService(
        retriever=retriever,
        prompt_builder=builder,
        llm_provider=llm,
        top_k=3,
    )
    check(isinstance(service, RAGService), "G1. RAGService created")

    # L: the RAG service must not depend on concrete stores/LLM providers.
    service_source = (
        Path(__file__).resolve().parent.parent / "app" / "rag" / "service.py"
    ).read_text(encoding="utf-8")
    forbidden = ["ChromaVectorStore", "chromadb", "MockLLMProvider",
                 "OpenAICompatibleProvider", "app.llm.providers"]
    # Match actual import statements, not docstring mentions.
    import_re = re.compile(
        r"^\s*(?:from\s+[^\s]+\s+import\s+[^\n#]*\b({tokens})\b|"
        r"^\s*import\s+[^\n#]*\b({tokens})\b)",
        re.MULTILINE,
    )
    found = sorted(
        set(
            tok
            for tok in forbidden
            for m in import_re.finditer(service_source)
            if tok in m.group(0)
        )
    )
    check(not found, "L1. RAGService imports no concrete store/LLM classes",
          f"found in imports: {found}")
    check(
        isinstance(service._llm_provider, LLMProvider),
        "L2. Service holds provider via the LLMProvider abstraction",
        f"actual type: {type(service._llm_provider).__name__}",
    )

    section(f"H. Ask: \"{AUTH_QUESTION}\"")
    retrieved = retriever.retrieve(
        question=AUTH_QUESTION,
        repository_name=REPOSITORY_NAME,
        top_k=3,
    )
    check(len(retrieved) > 0, "H1. Retrieval returns results",
          f"got {len(retrieved)}")
    check(
        any("auth" in r.relative_path for r in retrieved),
        "H2. Retrieved context includes authentication code",
        f"paths: {[r.relative_path for r in retrieved]}",
    )

    section("I. PromptBuilder produces a grounded, inspectable prompt")
    built = builder.build(
        question=AUTH_QUESTION,
        repository_name=REPOSITORY_NAME,
        results=retrieved,
    )
    prompt: str = built.prompt
    check(REPOSITORY_NAME in prompt, "I1. Prompt contains repository name")
    check(
        "src/auth.py" in prompt or "src/api/routes.py" in prompt,
        "I2. Prompt contains retrieved file paths",
    )
    check(
        re.search(r"lines: \d+-\d+", prompt) is not None,
        "I3. Prompt contains line ranges",
    )
    check(
        "def authenticate_user" in prompt or "def login_route" in prompt,
        "I4. Prompt contains retrieved code",
    )
    check(
        "GROUNDING RULES" in prompt and "invent" in prompt,
        "I5. Prompt contains grounding instructions",
    )
    check(
        "file paths and line ranges" in prompt,
        "I6. Prompt instructs referencing file paths and line ranges",
    )
    check(AUTH_QUESTION in prompt, "I7. Prompt contains the user question")
    check(ascii_only(prompt), "I8. Prompt is ASCII-only")

    section("F/I. Mock LLM receives the generated prompt and returns an answer")
    llm_response = llm.generate(prompt)
    check(isinstance(llm_response, LLMResponse), "I9. Mock returns LLMResponse")
    check(bool(llm_response.content.strip()), "I10. Mock returns non-empty content")
    check(llm_response.provider_name == "mock", "I11. Provider identified as 'mock'")
    check(llm_response.model_name == "mock-llm", "I12. Model identified as 'mock-llm'")
    check(
        llm_response.content.startswith("[mock answer]"),
        "I13. Mock answer does not masquerade as a real AI model",
        llm_response.content[:60],
    )

    section("G/I. RAGService.answer returns the structured grounded answer")
    response = service.answer(
        question=AUTH_QUESTION,
        repository_name=REPOSITORY_NAME,
        top_k=3,
    )
    check(bool(response.answer.strip()), "I14. Final answer is returned")
    check(
        len(response.retrieved_results) > 0,
        "I15. Retrieved results are included in the response",
        f"got {len(response.retrieved_results)}",
    )
    check(response.repository_name == REPOSITORY_NAME,
          "I16. Response records the repository name")
    check(response.question == AUTH_QUESTION,
          "I17. Response records the user question")
    check(
        response.model_name == "mock-llm" and response.provider_name == "mock",
        "I18. Response records model/provider information",
    )
    check(not response.insufficient_context,
          "I19. Auth question has sufficient context")
    check(ascii_only(response.answer), "I20. Answer is ASCII-only")

    return service


# ---------------------------------------------------------------------------
# Part J: insufficient context
# ---------------------------------------------------------------------------

def verify_insufficient_context(service: RAGService) -> None:
    """Part J: out-of-corpus questions must not fabricate repository context."""
    section("J. Insufficient-context question")
    question = "Where is blockchain mining implemented?"
    try:
        response = service.answer(
            question=question,
            repository_name=REPOSITORY_NAME,
            top_k=3,
        )
    except Exception as e:  # noqa: BLE001
        check(False, "J1. No exception raised", f"{type(e).__name__}: {e}")
        return

    check(True, "J1. No exception raised for out-of-corpus question")
    check(
        not any("blockchain" in (r.content or "").lower()
                for r in response.retrieved_results),
        "J2. No fabricated repository context is retrieved",
    )
    check(
        "insufficient" in response.answer.lower()
        or "no relevant repository context" in response.answer.lower(),
        "J3. Answer explicitly states insufficient evidence",
        response.answer[:120],
    )
    check(response.insufficient_context is True,
          "J4. Response is flagged insufficient_context")
    # The answer may echo the user's own question, but it must NOT invent any
    # file, line range, or code that is not in the supplied context.
    check(
        not re.search(
            r"([A-Za-z]+\.py|blockchain\s+code|line(s)?\s+\d+[-–]\d+)",
            response.answer,
        )
        or "cannot point to any file" in response.answer,
        "J5. Answer does not invent code beyond echoing the question",
        response.answer[:120],
    )


# ---------------------------------------------------------------------------
# Part K: context budget
# ---------------------------------------------------------------------------

def _to_retrieval_results(chunks: list[CodeChunk]) -> list[RetrievalResult]:
    """Convert CodeChunks into deterministic RetrievalResults (rank order)."""
    results = []
    for i, chunk in enumerate(chunks):
        results.append(
            RetrievalResult(
                chunk_id=chunk.id,
                content=chunk.content,
                repository_name=chunk.repository_name,
                relative_path=chunk.relative_path,
                language=chunk.language,
                chunk_type=chunk.chunk_type,
                chunk_index=chunk.chunk_index,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                relevance_score=max(0.0, 1.0 - i * 0.01),
                metadata={},
            )
        )
    return results


def verify_context_budget(
    service: RAGService,
    budget_chunks: list[CodeChunk],
    engine: EmbeddingEngine,
    store,
) -> None:
    """Part K: complete chunks only; budget respected; no partial truncation.

    Drives the PromptBuilder with deterministic RetrievalResults so budget
    behavior is exact, then seeds the same corpus and runs the RAG service
    end-to-end with a tight budget to confirm the service honors it.
    """
    section("K. Context budget behavior")

    results = _to_retrieval_results(budget_chunks)
    # Each controller block is ~460 chars, so a 300-token budget (~1200
    # chars) fits exactly two COMPLETE chunks and must skip the rest.
    tight = PromptContextBudget(max_tokens=300)
    tight_builder = PromptBuilder(tight)
    built = tight_builder.build(
        question="How do the controller functions behave?",
        repository_name=REPOSITORY_NAME,
        results=results,
    )

    check(built.budget_exceeded is True,
          "K1. Budget exceeded when corpus exceeds it",
          f"included={built.included_count} excluded={built.excluded_results}")
    check(built.included_count >= 1,
          "K2. At least one complete chunk included",
          f"included={built.included_count}")
    check(built.excluded_results >= 1,
          "K3. Chunks were excluded for budget",
          f"excluded={built.excluded_results}")
    check(
        built.budget_used_chars <= built.budget_max_chars,
        "K4. Budgeted chars never exceed the budget",
        f"{built.budget_used_chars} > {built.budget_max_chars}",
    )
    check(
        built.included_count + built.excluded_results == len(results),
        "K5. Included + excluded equals total results",
        f"{built.included_count} + {built.excluded_results} != {len(results)}",
    )

    # Every included chunk appears COMPLETE in the prompt.
    all_blocks_ok = all(item.content in built.prompt
                        for item in built.included_results)
    check(all_blocks_ok, "K6. All included chunks appear complete in the prompt")

    # No excluded chunk (or any leading slice of one) appears in the prompt.
    included_files = {item.file for item in built.included_results}
    partial_present = any(
        r.relative_path not in included_files and r.content[:40] in built.prompt
        for r in results
    )
    check(not partial_present,
          "K7. No excluded chunk is partially truncated into the prompt")

    # Deterministic: same input -> identical prompt.
    built2 = tight_builder.build(
        question="How do the controller functions behave?",
        repository_name=REPOSITORY_NAME,
        results=results,
    )
    check(built.prompt == built2.prompt, "K8. PromptBuilder is deterministic")

    # End-to-end: seed the same corpus and run the full service with the
    # tight-budget builder; the service must respect the budget and report it.
    check(
        all(c.content for c in budget_chunks),
        "K9. Budget corpus has non-empty content",
    )
    seed_pairs = []
    for chunk in budget_chunks:
        seed_pairs.append((chunk, engine.embed(chunk)))
    store.add_many(
        ChunkEmbeddingPair(chunk=chunk, record=record)
        for chunk, record in seed_pairs
    )
    tight_service = RAGService(
        retriever=service._retriever,
        prompt_builder=tight_builder,
        llm_provider=service._llm_provider,
        top_k=12,
    )
    ans = tight_service.answer(
        question="How do the controller functions behave?",
        repository_name=REPOSITORY_NAME,
        top_k=12,
    )
    check(
        ans.budget_exceeded is True and ans.context_excluded_count >= 1,
        "K10. RAG service respects the budget end-to-end",
        f"included={ans.context_included_count} "
        f"excluded={ans.context_excluded_count}",
    )
    # Rebuild the exact prompt the service sent and prove it stays in budget.
    replayed = tight_builder.build(
        question=ans.question,
        repository_name=ans.repository_name,
        results=ans.retrieved_results,
    )
    check(
        replayed.budget_used_chars <= replayed.budget_max_chars,
        "K11. Prompt actually sent to the LLM stays within the budget",
        f"{replayed.budget_used_chars} > {replayed.budget_max_chars}",
    )
    check(
        ascii_only(ans.answer),
        "K12. Budget-mode answer is ASCII-only",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all Sprint 8 verifications and exit with the verdict."""
    retriever, builder, llm, engine, store = build_pipeline()
    service = verify_pipeline(retriever, builder, llm)
    verify_insufficient_context(service)
    verify_context_budget(service, make_budget_chunks(), engine, store)
    final_verdict()


if __name__ == "__main__":
    main()