"""PromptBuilder: turns retrieved code into a structured, grounded prompt.

The PromptBuilder is responsible for exactly one thing: converting a user
question, a repository name, and a list of RetrievalResult objects into a
deterministic, inspectable LLM prompt. It NEVER calls an LLM; generation is
the job of an LLMProvider (app.llm).

Grounding design (anti-hallucination):
    The prompt explicitly instructs the model to answer ONLY from the
    supplied repository context, to never invent files, functions, classes,
    APIs, configuration, or behavior, and to state when the supplied evidence
    is insufficient. Each context block is self-describing: rank, relative
    file path, language, chunk type, start/end lines, relevance score, and
    content. The model is told to reference file paths and line ranges when
    making claims.

Context budget design:
    The builder must not blindly include unlimited retrieved chunks. It uses
    PromptContextBudget, a token budget converted to an approximate character
    budget via a documented heuristic (~4 chars/token), so no tokenizer
    dependency is introduced. Chunks are included in rank order; a chunk is
    only ever included COMPLETE — if the next chunk would exceed the budget,
    it is skipped (and reported as excluded) rather than truncated in the
    middle. A chunk too large for the entire budget is never partially
    included. The prompt stays deterministic and easy to inspect.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Tuple

from app.prompts.models import BuiltPrompt, PromptContextBudget, PromptContextItem

# Imported under TYPE_CHECKING only to break the import cycle
# prompts.builder -> rag.models -> prompts.models. The builder only needs
# RetrievalResult for annotations; it accesses result attributes via duck
# typing at runtime.
if TYPE_CHECKING:
    from app.rag.models import RetrievalResult

logger = logging.getLogger(__name__)

# Template for one retrieved code block. Kept as a module-level constant so
# the exact prompt shape is easy to inspect and test.
_SYSTEM_INSTRUCTIONS = (
    "You are CodePilot, an AI assistant that analyzes repository code. "
    "You answer questions about the code supplied below."
)

_GROUNDING_INSTRUCTIONS = (
    "GROUNDING RULES (follow these strictly):\n"
    "1. Use ONLY the repository context supplied in the RETRIEVED CONTEXT "
    "section below. Do not use any outside knowledge of this repository.\n"
    "2. Do not invent files, functions, classes, APIs, configuration, or "
    "behavior that are not present in the supplied context.\n"
    "3. If the supplied context is insufficient to answer the question, say "
    "so explicitly. Do not pretend to know code that was not supplied.\n"
    "4. When you make a claim about code, reference the relevant file path "
    "and line range from the supplied context (for example: "
    "src/auth.py lines 10-25).\n"
    "5. Never invent repository details."
)


class PromptBuilder:
    """Builds grounded LLM prompts from retrieved code chunks.

    Attributes:
        budget: The context budget applied to the retrieved-context section.

    The builder is stateless between calls: each build() reads the current
    budget and produces a self-contained BuiltPrompt. It depends only on the
    RetrievalResult model (from app.rag) and the prompt models; it has no
    knowledge of any LLM provider.
    """

    def __init__(self, budget: PromptContextBudget | None = None):
        """Initialize the prompt builder.

        Args:
            budget: Context budget to use. If None, a default budget is
                created from configuration (LLM_CONTEXT_BUDGET_TOKENS env).
        """
        self.budget = budget or PromptContextBudget()
        logger.info(
            f"PromptBuilder initialized "
            f"(budget: {self.budget.max_tokens} tokens / "
            f"{self.budget.max_chars} chars)"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        question: str,
        repository_name: str,
        results: List[RetrievalResult],
    ) -> BuiltPrompt:
        """Build a grounded prompt from a question and retrieved results.

        Args:
            question: The user's question (validated by the RAG service).
            repository_name: Repository the question was asked about.
            results: Retrieved, relevance-ordered results (most relevant
                first). The first result is rank 1.

        Returns:
            A BuiltPrompt containing the prompt string and deterministic
            budget bookkeeping.

        Raises:
            ValueError: If question or repository_name is empty/whitespace.
        """
        question = self._validate_text(question, "question")
        repository_name = self._validate_text(repository_name, "repository_name")

        max_chars = self.budget.max_chars

        included: List[PromptContextItem] = []
        used_chars = 0
        excluded_count = 0

        for rank, result in enumerate(results, start=1):
            item = self._to_context_item(rank, result)
            _, block_chars = self._render_block(item)

            # Only ever include a COMPLETE chunk. If this chunk does not fit
            # (either alone or on top of what is already included), skip it
            # and stop: skipping later chunks cannot make the budget fit
            # again, and truncating mid-chunk is explicitly forbidden.
            if used_chars + block_chars > max_chars:
                excluded_count = len(results) - (rank - 1)
                logger.info(
                    f"Context budget reached: skipped {excluded_count} result(s) "
                    f"to stay within {max_chars} chars"
                )
                break

            included.append(item)
            used_chars += block_chars

        prompt = self._assemble(question, repository_name, included)

        built = BuiltPrompt(
            prompt=prompt,
            repository_name=repository_name,
            question=question,
            total_results=len(results),
            included_results=included,
            excluded_results=excluded_count,
            total_chars=len(prompt),
            budget_max_tokens=self.budget.max_tokens,
            budget_max_chars=max_chars,
            budget_used_chars=used_chars,
            budget_exceeded=bool(results) and len(included) < len(results),
        )

        logger.info(
            f"Built prompt: {len(included)}/{len(results)} chunks included, "
            f"{excluded_count} excluded, {used_chars}/{max_chars} budget chars used"
        )
        return built

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _render_block(self, item: PromptContextItem) -> Tuple[str, int]:
        """Render one context item into its prompt block.

        Args:
            item: The context item to render.

        Returns:
            A (block_text, block_chars) tuple. block_chars is the size the
            block contributes to the budget, so budget accounting is
            consistent with what the prompt actually contains.
        """
        header = (
            f"[{item.rank}] file: {item.file} "
            f"| language: {item.language} "
            f"| chunk_type: {item.chunk_type} "
            f"| lines: {item.start_line}-{item.end_line} "
            f"| relevance: {item.relevance_score:.4f}"
        )
        block = (
            f"{header}\n"
            f"{''.join('=' for _ in range(len(header)))}\n"
            f"{item.content}"
        )
        return block, len(block)

    def _assemble(
        self,
        question: str,
        repository_name: str,
        included: List[PromptContextItem],
    ) -> str:
        """Assemble the full prompt string from its parts.

        The fixed overhead (system + grounding instructions, repository,
        question) is not charged to the context budget; only the RETRIEVED
        CONTEXT section is budgeted.

        Args:
            question: The user question.
            repository_name: Repository the question was asked about.
            included: The context items that fit within the budget.

        Returns:
            The complete prompt string.
        """
        sections: List[str] = [
            _SYSTEM_INSTRUCTIONS,
            "",
            _GROUNDING_INSTRUCTIONS,
            "",
            f"REPOSITORY:\n{repository_name}",
            "",
        ]

        if included:
            sections.append("RETRIEVED CONTEXT:")
            for item in included:
                block, _ = self._render_block(item)
                sections.append(block)
        else:
            sections.append(
                "RETRIEVED CONTEXT:\n"
                "(no relevant repository context was retrieved for this question)"
            )

        sections.extend(
            [
                "",
                "USER QUESTION:",
                question,
                "",
                "INSTRUCTIONS:",
                "Answer the user's question using only the repository context "
                "above. Reference file paths and line ranges when you make "
                "claims. If the supplied context is insufficient, say so "
                "explicitly and do not invent code.",
            ]
        )

        return "\n".join(sections)

    @staticmethod
    def _to_context_item(rank: int, result: RetrievalResult) -> PromptContextItem:
        """Convert a RetrievalResult into a prompt-facing context item.

        Args:
            rank: 1-based rank of the result.
            result: The retrieved result.

        Returns:
            A PromptContextItem with normalized fields.
        """
        return PromptContextItem(
            rank=rank,
            file=result.relative_path or "unknown",
            language=result.language or "unknown",
            chunk_type=result.chunk_type.value,
            start_line=result.start_line,
            end_line=result.end_line,
            relevance_score=result.relevance_score,
            content=result.content or "",
        )

    @staticmethod
    def _validate_text(value: str, name: str) -> str:
        """Validate a required text field.

        Args:
            value: The raw value.
            name: Field name for error messages.

        Returns:
            The trimmed value.

        Raises:
            ValueError: If the value is not a non-empty string.
        """
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a string, got {type(value).__name__}")
        trimmed = value.strip()
        if not trimmed:
            raise ValueError(f"{name} must not be empty or whitespace")
        return trimmed


__all__ = ["PromptBuilder"]
