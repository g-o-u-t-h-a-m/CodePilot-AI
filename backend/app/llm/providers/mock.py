"""Deterministic mock LLM provider for testing.

MockLLMProvider implements the LLMProvider interface but is NOT a real AI
model. It parses the prompt built by the PromptBuilder and returns a
predictable, deterministic answer that reflects the supplied context. This
lets the entire RAG pipeline be tested end-to-end with no API calls, no
API key, and no cost (the project's 0-budget requirement).

The mock never fabricates repository context: if the prompt contains no
retrieved context blocks, it returns an explicit insufficient-evidence
answer, mirroring the grounding behavior the prompt instructs a real model
to follow.
"""

import logging
import re
from typing import List, Tuple

from app.llm.models import LLMResponse
from app.llm.provider import LLMProvider

logger = logging.getLogger(__name__)


class MockLLMProvider(LLMProvider):
    """Deterministic mock LLM provider.

    provider_name is 'mock' and model_name is 'mock-llm', so every response
    is unambiguously identifiable as coming from the mock rather than a real
    model. The generated text is derived solely from the prompt's REPOSITORY,
    RETRIEVED CONTEXT, and USER QUESTION sections.
    """

    PROVIDER_NAME = "mock"
    MODEL_NAME = "mock-llm"

    # Header of each retrieved context block, e.g.:
    # "[1] file: src/auth.py | language: Python | chunk_type: function
    #  | lines: 10-25 | relevance: 0.8700"
    _BLOCK_RE = re.compile(
        r"\[\d+\] file: (?P<file>[^ ]+) \| language: (?P<lang>[^ ]+) "
        r"\| chunk_type: (?P<type>[^ ]+) "
        r"\| lines: (?P<start>\d+)-(?P<end>\d+) "
        r"\| relevance: (?P<score>[\d.]+)"
    )

    def __init__(self):
        """Initialize the mock provider.

        The mock requires no configuration, no API key, and no network.
        """
        logger.info(f"Initializing {self.__class__.__name__} (mock, no API)")

    def generate(self, prompt: str) -> LLMResponse:
        """Generate a deterministic response from the prompt.

        Args:
            prompt: The prompt produced by the PromptBuilder.

        Returns:
            An LLMResponse whose content is derived from the prompt's own
            context, clearly identified as produced by the mock provider.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        question = self._extract_question(prompt)
        repository_name = self._extract_repository(prompt)
        blocks = self._extract_blocks(prompt)

        if not blocks:
            content = (
                f"I have insufficient evidence to answer the question "
                f"{question!r} about the repository '{repository_name}'. "
                "No relevant repository context was retrieved, so I cannot "
                "point to any file or line range. Please narrow the question "
                "or re-index the repository. (mock provider)"
            )
        else:
            lines = [
                f"- {file} lines {start}-{end} ({lang}, {chunk_type}, "
                f"relevance {score:.4f})"
                for (file, start, end, lang, chunk_type, score) in blocks
            ]
            content = (
                f"[mock answer] For the question {question!r} about the "
                f"repository '{repository_name}', the relevant code is:\n"
                + "\n".join(lines)
                + "\nThis deterministic answer references the supplied "
                "context only; it does not invent any code. (mock provider)"
            )

        response = LLMResponse(
            content=content,
            model_name=self.MODEL_NAME,
            provider_name=self.PROVIDER_NAME,
            metadata={
                "mock": True,
                "prompt_chars": len(prompt),
                "blocks": len(blocks),
            },
        )
        logger.info(
            f"Mock provider returned {len(content)} chars "
            f"({len(blocks)} context blocks)"
        )
        return response

    def get_provider_name(self) -> str:
        """Get the provider name ('mock')."""
        return self.PROVIDER_NAME

    def get_model_name(self) -> str:
        """Get the model name ('mock-llm')."""
        return self.MODEL_NAME

    # ------------------------------------------------------------------
    # Internal helpers: deterministic prompt parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_question(prompt: str) -> str:
        """Extract the user question from the USER QUESTION section.

        Args:
            prompt: The built prompt.

        Returns:
            The question text, or a fallback if the section is missing.
        """
        match = re.search(r"USER QUESTION:\n(.+)", prompt)
        if not match:
            return "(unknown)"
        return match.group(1).strip()

    @staticmethod
    def _extract_repository(prompt: str) -> str:
        """Extract the repository name from the REPOSITORY section.

        Args:
            prompt: The built prompt.

        Returns:
            The repository name, or a fallback if the section is missing.
        """
        match = re.search(r"REPOSITORY:\n([^\n]+)", prompt)
        if not match:
            return "(unknown)"
        return match.group(1).strip()

    @classmethod
    def _extract_blocks(cls, prompt: str) -> List[Tuple[str, int, int, str, str, float]]:
        """Extract all retrieved context blocks from the prompt.

        Args:
            prompt: The built prompt.

        Returns:
            List of (file, start, end, language, chunk_type, score) tuples in
            rank order, empty if the prompt has no retrieved context.
        """
        blocks: List[Tuple[str, int, int, str, str, float]] = []
        for match in cls._BLOCK_RE.finditer(prompt):
            blocks.append(
                (
                    match.group("file"),
                    int(match.group("start")),
                    int(match.group("end")),
                    match.group("lang"),
                    match.group("type"),
                    float(match.group("score")),
                )
            )
        return blocks


__all__ = ["MockLLMProvider"]