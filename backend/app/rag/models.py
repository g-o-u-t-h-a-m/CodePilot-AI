"""Data models for the retrieval (RAG) layer.

This module defines application-facing models for semantic retrieval,
independent of any specific vector database. The Retriever produces
RetrievalResult objects that represent a single relevant code chunk
found by querying the vector store with a natural-language question.

The model deliberately mirrors the metadata preserved by the vector
store (repository_name, relative_path, language, etc.) so that later
sprints (prompt building, LLM generation) can consume results without
knowing anything about ChromaDB or vector storage internals.

Sprint 8 adds RagResponse, the structured result of the full RAG
pipeline (retrieval + prompt building + generation).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.chunking.models import ChunkType

# Imported under TYPE_CHECKING only to break the import cycle
# prompts.builder -> rag.models -> prompts.models. The RagResponse.from_components
# factory receives llm_response and built_prompt as arguments, so it never needs
# these classes at runtime.
if TYPE_CHECKING:
    from app.llm.models import LLMResponse
    from app.prompts.models import BuiltPrompt


class RetrievalResult(BaseModel):
    """A single code chunk retrieved in response to a user question.

    This is the unit of retrieval: one relevant chunk, its source
    location, and a normalized relevance score. It is constructed by
    the Retriever from a vector store SimilarityResult, so it never
    exposes raw database distance semantics.

    Attributes:
        chunk_id: Identifier of the retrieved code chunk
        content: The chunk content (source code / text)
        repository_name: Repository the chunk belongs to
        relative_path: Path of the source file, relative to repo root
        language: Programming language of the chunk
        chunk_type: Type of chunk (function, class, module, ...)
        chunk_index: Index of the chunk within its source file (0-based)
        start_line: Starting line number of the chunk (1-based)
        end_line: Ending line number of the chunk (1-based)
        relevance_score: Normalized relevance in [0, 1]; higher = more relevant
        metadata: Full metadata preserved from the vector store
    """

    chunk_id: str = Field(
        ...,
        description="Identifier of the retrieved code chunk"
    )
    content: str = Field(
        ...,
        description="The chunk content (source code or text)"
    )
    repository_name: str = Field(
        ...,
        description="Repository the chunk belongs to"
    )
    relative_path: str = Field(
        ...,
        description="Path of the source file relative to repository root"
    )
    language: str = Field(
        ...,
        description="Programming language of the chunk"
    )
    chunk_type: ChunkType = Field(
        ...,
        description="Type of chunk (function, class, module, ...)"
    )
    chunk_index: int = Field(
        ...,
        description="Index of the chunk within its source file (0-based)",
        ge=0
    )
    start_line: int = Field(
        ...,
        description="Starting line number of the chunk (1-based)",
        ge=1
    )
    end_line: int = Field(
        ...,
        description="Ending line number of the chunk (1-based)",
        ge=1
    )
    relevance_score: float = Field(
        ...,
        description="Normalized relevance in [0, 1]; higher = more relevant",
        ge=0.0,
        le=1.0
    )
    metadata: Dict[str, object] = Field(
        default_factory=dict,
        description="Full metadata preserved from the vector store"
    )

    @field_validator("chunk_type", mode="before")
    @classmethod
    def coerce_chunk_type(cls, value) -> ChunkType:
        """Coerce a stored chunk type string into the ChunkType enum.

        ChromaDB persists the ChunkType as its string value (e.g.
        "function"), so this validator accepts both the enum member
        and its string form.

        Args:
            value: Raw chunk_type value (ChunkType or str)

        Returns:
            The validated ChunkType member

        Raises:
            ValueError: If the value is not a valid chunk type
        """
        if isinstance(value, ChunkType):
            return value
        if isinstance(value, str):
            return ChunkType(value)
        raise ValueError(
            f"Invalid chunk_type value: {value!r} "
            f"(expected ChunkType or str)"
        )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chunk_id": "550e8400-e29b-41d4-a716-446655440000",
                "content": "def authenticate_user():\n    pass",
                "repository_name": "demo-repo",
                "relative_path": "src/auth.py",
                "language": "Python",
                "chunk_type": "function",
                "chunk_index": 0,
                "start_line": 10,
                "end_line": 25,
                "relevance_score": 0.87,
                "metadata": {
                    "repository_name": "demo-repo",
                    "relative_path": "src/auth.py",
                    "language": "Python",
                    "chunk_type": "function",
                    "chunk_index": 0,
                    "start_line": 10,
                    "end_line": 25,
                    "content_hash": "abc123def456",
                    "model_name": "BAAI/bge-small-en-v1.5"
                }
            }
        }
    )


class RagResponse(BaseModel):
    """Structured result of the end-to-end RAG pipeline.

    Sprint 8: produced by RAGService.answer(). Contains the grounded answer
    plus full provenance: the question, repository, retrieved results used to
    ground the answer, provider/model metadata, and prompt-builder budget
    bookkeeping.

    Attributes:
        answer: The grounded natural-language answer produced by the LLM.
        repository_name: Repository the question was asked about.
        question: The user's question.
        retrieved_results: The retrieval results used to ground the answer.
        model_name: Model that produced the answer, if known.
        provider_name: Provider that produced the answer.
        prompt_chars: Length of the prompt sent to the LLM.
        retrieved_count: Number of retrieved results passed to the builder.
        context_included_count: Results that fit the budget and were included.
        context_excluded_count: Results dropped because they exceeded budget.
        budget_exceeded: True when not all retrieved results fit the budget.
        insufficient_context: True when no relevant context was retrieved and
            the service returned an explicit insufficient-context response
            (no fabricated context).
        metadata: Additional pipeline metadata (e.g. LLM provider metadata).
    """

    answer: str = Field(..., description="The grounded answer text")
    repository_name: str = Field(..., description="Repository queried")
    question: str = Field(..., description="The user's question")
    retrieved_results: List[RetrievalResult] = Field(
        default_factory=list,
        description="Retrieved results used to ground the answer",
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Model that produced the answer",
    )
    provider_name: Optional[str] = Field(
        default=None,
        description="Provider that produced the answer",
    )
    prompt_chars: int = Field(
        default=0,
        description="Prompt length in characters",
        ge=0,
    )
    retrieved_count: int = Field(
        default=0,
        description="Number of retrieved results passed to the builder",
        ge=0,
    )
    context_included_count: int = Field(
        default=0,
        description="Results included in the prompt (within budget)",
        ge=0,
    )
    context_excluded_count: int = Field(
        default=0,
        description="Results dropped because they exceeded the budget",
        ge=0,
    )
    budget_exceeded: bool = Field(
        default=False,
        description="True when not all retrieved results fit the budget",
    )
    insufficient_context: bool = Field(
        default=False,
        description="True when no relevant context was retrieved",
    )
    metadata: Dict[str, object] = Field(
        default_factory=dict,
        description="Additional pipeline metadata",
    )

    @classmethod
    def from_components(
        cls,
        answer: str,
        repository_name: str,
        question: str,
        retrieved_results: List[RetrievalResult],
        llm_response: LLMResponse,
        built_prompt: BuiltPrompt,
    ) -> "RagResponse":
        """Build a RagResponse from pipeline components.

        Args:
            answer: The generated answer text.
            repository_name: Repository queried.
            question: The user's question.
            retrieved_results: Retrieved results used to ground the answer.
            llm_response: The LLM response (provider/model metadata).
            built_prompt: The built prompt (budget bookkeeping).

        Returns:
            A fully populated RagResponse.
        """
        return cls(
            answer=answer,
            repository_name=repository_name,
            question=question,
            retrieved_results=retrieved_results,
            model_name=llm_response.model_name,
            provider_name=llm_response.provider_name,
            prompt_chars=built_prompt.total_chars,
            retrieved_count=built_prompt.total_results,
            context_included_count=built_prompt.included_count,
            context_excluded_count=built_prompt.excluded_results,
            budget_exceeded=built_prompt.budget_exceeded,
            insufficient_context=not retrieved_results,
            metadata={"llm_metadata": llm_response.metadata},
        )

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "answer": "User authentication is handled in src/auth.py lines 10-25.",
                "repository_name": "demo-repo",
                "question": "Where is user authentication handled?",
                "retrieved_results": [],
                "model_name": "mock-llm",
                "provider_name": "mock",
                "prompt_chars": 2048,
                "retrieved_count": 2,
                "context_included_count": 2,
                "context_excluded_count": 0,
                "budget_exceeded": False,
                "insufficient_context": False,
                "metadata": {},
            }
        }
    )


__all__ = ["RetrievalResult", "RagResponse"]
