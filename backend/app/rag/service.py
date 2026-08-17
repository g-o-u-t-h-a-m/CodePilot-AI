"""RAG service: orchestrates retrieval, prompt building, and generation.

RAGService is the generation-layer entry point for the RAG pipeline. It
orchestrates:

    1. Validate the user question
    2. Call the Retriever to get RetrievalResult[]
    3. Pass results to the PromptBuilder to build a grounded prompt
    4. Send the prompt to an LLMProvider to generate an answer
    5. Return a structured RagResponse

Dependency Inversion:
    The service depends ONLY on abstractions: Retriever (Sprint 7),
    PromptBuilder, and LLMProvider. It never imports ChromaDB,
    ChromaVectorStore, or any concrete LLM implementation. Each dependency is
    injected through the constructor, so the vector database and the LLM can
    be swapped without touching this module.

No-fabrication handling:
    If no relevant retrieval results exist, the service does NOT fabricate
    context. It builds a prompt that explicitly states no context was
    retrieved (the PromptBuilder emits a "no relevant repository context"
    placeholder), sends it to the LLM, and marks the response as
    insufficient_context=True. A real grounded model will answer that it has
    insufficient evidence; the MockLLMProvider does exactly this for tests.
"""

import logging
import os
from typing import List, Optional

from app.llm.provider import LLMProvider
from app.prompts.builder import PromptBuilder
from app.rag.models import RagResponse, RetrievalResult
from app.rag.retriever import Retriever

logger = logging.getLogger(__name__)


class RAGService:
    """Orchestrates the RAG pipeline from question to grounded answer.

    Responsibilities:
        - Validate the user question with a meaningful error message
        - Delegate retrieval to the injected Retriever
        - Delegate prompt building to the injected PromptBuilder
        - Delegate generation to the injected LLMProvider
        - Return a structured RagResponse with full provenance

    SOLID notes:
        - Single Responsibility: only orchestrates the RAG pipeline
        - Open/Closed: extendable without modifying its dependencies
        - Dependency Inversion: depends on Retriever, PromptBuilder,
          LLMProvider abstractions, never on ChromaDB or a concrete LLM
    """

    # Default minimum relevance score for a result to count as "relevant"
    # context. Rationale: top-k retrieval always returns k results (even for
    # unrelated questions the vector store returns the k least-bad matches),
    # so the service needs an explicit, documented relevance gate to decide
    # when recovered context is genuinely usable. The BGE provider normalizes
    # embeddings over a cosine metric where results meaningfully related to
    # the question typically score >= ~0.55, while unrelated matches fall
    # below it. A result at or above this score grounds the answer; if NONE
    # reach it, the service treats the context as insufficient (no
    # fabrication). Configurable via the RAG_MIN_RELEVANCE environment
    # variable.
    DEFAULT_MIN_RELEVANCE = 0.55
    _MIN_RELEVANCE_ENV = "RAG_MIN_RELEVANCE"

    def __init__(
        self,
        retriever: Retriever,
        prompt_builder: PromptBuilder,
        llm_provider: LLMProvider,
        top_k: int = 5,
        min_relevance: Optional[float] = None,
    ):
        """Initialize the RAG service with its dependencies.

        Args:
            retriever: Existing Retriever (Sprint 7) used to fetch relevant
                code chunks.
            prompt_builder: PromptBuilder used to build the grounded prompt.
            llm_provider: LLMProvider used to generate the answer. The mock
                provider enables 0-cost local testing.
            top_k: Default number of retrieval results to fetch.
            min_relevance: Minimum normalized relevance score for a retrieved
                result to count as usable context. If None, read from the
                RAG_MIN_RELEVANCE environment variable, falling back to
                DEFAULT_MIN_RELEVANCE.

        Raises:
            TypeError: If a dependency is missing or of the wrong type.
        """
        if retriever is None:
            raise TypeError("retriever is required")
        if prompt_builder is None:
            raise TypeError("prompt_builder is required")
        if llm_provider is None:
            raise TypeError("llm_provider is required")
        if not isinstance(retriever, Retriever):
            raise TypeError(
                f"retriever must be a Retriever, "
                f"got {type(retriever).__name__}"
            )
        if not isinstance(prompt_builder, PromptBuilder):
            raise TypeError(
                f"prompt_builder must be a PromptBuilder, "
                f"got {type(prompt_builder).__name__}"
            )
        if not isinstance(llm_provider, LLMProvider):
            raise TypeError(
                f"llm_provider must implement LLMProvider, "
                f"got {type(llm_provider).__name__}"
            )
        self._validate_top_k(top_k)

        self._retriever = retriever
        self._prompt_builder = prompt_builder
        self._llm_provider = llm_provider
        self._top_k = top_k
        self._min_relevance = self._resolve_min_relevance(min_relevance)

        logger.info(
            f"RAGService initialized "
            f"(retriever: {retriever.__class__.__name__}, "
            f"prompt_builder: {prompt_builder.__class__.__name__}, "
            f"llm_provider: {llm_provider.get_provider_name()}, "
            f"top_k: {top_k}, min_relevance: {self._min_relevance})"
        )

    def answer(
        self,
        question: str,
        repository_name: str,
        top_k: Optional[int] = None,
    ) -> RagResponse:
        """Run the full RAG pipeline for a question.

        Args:
            question: Natural-language user question (non-empty).
            repository_name: Repository to search within (non-empty).
            top_k: Maximum number of retrieval results to fetch. If None,
                uses the service's configured default.

        Returns:
            A RagResponse containing the grounded answer and full provenance.

        Raises:
            ValueError: If question/repository_name is empty/whitespace, or
                top_k is not a positive integer.
            RuntimeError: If retrieval, prompt building, or generation fails.
        """
        question = self._validate_question(question)
        repository_name = self._validate_repository_name(repository_name)
        resolved_top_k = self._resolve_top_k(top_k)

        logger.info(
            f"Answering question for repository '{repository_name}': {question!r}"
        )

        # 1. Retrieve relevant code chunks (existing Retriever).
        retrieved: List[RetrievalResult] = self._retriever.retrieve(
            question=question,
            repository_name=repository_name,
            top_k=resolved_top_k,
        )
        logger.info(f"Retrieved {len(retrieved)} results")

        # 2. Keep only results whose relevance is high enough to ground an
        #    answer without fabrication. A related question's best matches
        #    score well above the gate; an unrelated question's matches fall
        #    below it, so no context is forced into the prompt.
        relevant = [
            result
            for result in retrieved
            if result.relevance_score >= self._min_relevance
        ]
        if len(relevant) < len(retrieved):
            logger.info(
                f"Relevance gate kept {len(relevant)}/{len(retrieved)} results"
            )

        # 3. Build a grounded prompt. If no relevant context exists, the
        #    prompt explicitly states there is no relevant context; nothing
        #    is fabricated.
        built = self._prompt_builder.build(
            question=question,
            repository_name=repository_name,
            results=relevant,
        )
        logger.info(
            f"Built prompt: {len(built.prompt)} chars, "
            f"{built.included_count} context blocks included"
        )

        # 4. Generate a grounded answer with the injected LLM provider.
        llm_response = self._llm_provider.generate(built.prompt)
        logger.info(
            f"Generated answer ({len(llm_response.content)} chars, "
            f"provider: {llm_response.provider_name}, "
            f"model: {llm_response.model_name})"
        )

        # 5. Return the structured response.
        response = RagResponse.from_components(
            answer=llm_response.content,
            repository_name=repository_name,
            question=question,
            retrieved_results=relevant,
            llm_response=llm_response,
            built_prompt=built,
        )
        logger.info(
            f"RAG pipeline complete: insufficient_context={response.insufficient_context}"
        )
        return response

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_question(question: str) -> str:
        """Validate the question is non-empty text.

        Args:
            question: Raw question input.

        Returns:
            The trimmed question.

        Raises:
            ValueError: If the question is empty or whitespace-only.
        """
        if not isinstance(question, str):
            raise ValueError(
                f"question must be a string, got {type(question).__name__}"
            )
        trimmed = question.strip()
        if not trimmed:
            raise ValueError("question must not be empty or whitespace")
        return trimmed

    @staticmethod
    def _validate_repository_name(repository_name: str) -> str:
        """Validate the repository name is non-empty text.

        Args:
            repository_name: Raw repository name input.

        Returns:
            The trimmed repository name.

        Raises:
            ValueError: If the name is empty or whitespace-only.
        """
        if not isinstance(repository_name, str):
            raise ValueError(
                f"repository_name must be a string, "
                f"got {type(repository_name).__name__}"
            )
        trimmed = repository_name.strip()
        if not trimmed:
            raise ValueError("repository_name must not be empty or whitespace")
        return trimmed

    def _resolve_min_relevance(self, min_relevance: Optional[float]) -> float:
        """Resolve the relevance gate from argument, env, or default.

        Args:
            min_relevance: Explicit value, or None.

        Returns:
            The effective relevance threshold in [0, 1].

        Raises:
            ValueError: If the resolved value is not a number in [0, 1].
        """
        if min_relevance is None:
            raw = os.environ.get(self._MIN_RELEVANCE_ENV)
            if raw is not None:
                try:
                    min_relevance = float(raw)
                except ValueError:
                    logger.warning(
                        f"Invalid {self._MIN_RELEVANCE_ENV}={raw!r}; "
                        f"falling back to {self.DEFAULT_MIN_RELEVANCE}"
                    )
                    min_relevance = self.DEFAULT_MIN_RELEVANCE
        if min_relevance is None:
            min_relevance = self.DEFAULT_MIN_RELEVANCE

        if isinstance(min_relevance, bool) or not isinstance(min_relevance, (int, float)):
            raise ValueError(
                f"min_relevance must be a number, "
                f"got {type(min_relevance).__name__}"
            )
        if not 0.0 <= min_relevance <= 1.0:
            raise ValueError(
                f"min_relevance must be in [0, 1], got {min_relevance}"
            )
        return float(min_relevance)

    def _resolve_top_k(self, top_k: Optional[int]) -> int:
        """Resolve the effective top_k for this call.

        Args:
            top_k: Per-call value or None.

        Returns:
            The effective top_k.

        Raises:
            ValueError: If the value is not a positive integer.
        """
        value = self._top_k if top_k is None else top_k
        self._validate_top_k(value)
        return value

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        """Validate that top_k is a positive integer.

        Args:
            top_k: Requested result count.

        Raises:
            ValueError: If top_k is not a positive integer.
        """
        if isinstance(top_k, bool):
            raise ValueError(f"top_k must be a positive integer, got {top_k}")
        if not isinstance(top_k, int):
            raise ValueError(
                f"top_k must be an integer, got {type(top_k).__name__}"
            )
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")


__all__ = ["RAGService"]