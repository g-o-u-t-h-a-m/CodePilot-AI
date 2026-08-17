"""Data models for prompt building.

These models are application-facing and independent of any LLM provider.
They describe how a user question, a repository name, and retrieved
RetrievalResult objects are turned into a structured prompt, plus the
deterministic bookkeeping the RAG service and tests can inspect.
"""

import os
from typing import List

from pydantic import BaseModel, ConfigDict, Field

# Characters-per-token heuristic used to translate a token budget into an
# approximate character budget without a tokenizer dependency. ~4 characters
# per token is a widely cited approximation for English prose and source code.
DEFAULT_CHARS_PER_TOKEN = 4.0

# Default context budget in tokens: bounds how much retrieved code is included
# in a prompt. Overridable via the LLM_CONTEXT_BUDGET_TOKENS environment
# variable (see .env.example).
DEFAULT_CONTEXT_BUDGET_TOKENS = 2000


def _default_budget_tokens() -> int:
    """Read the default context budget (in tokens) from the environment.

    Falls back to DEFAULT_CONTEXT_BUDGET_TOKENS if the variable is unset,
    non-numeric, or not positive so an invalid value can never crash import.

    Returns:
        The configured budget in tokens.
    """
    raw = os.getenv("LLM_CONTEXT_BUDGET_TOKENS", str(DEFAULT_CONTEXT_BUDGET_TOKENS))
    try:
        value = int(raw)
    except ValueError:
        value = DEFAULT_CONTEXT_BUDGET_TOKENS
    return value if value > 0 else DEFAULT_CONTEXT_BUDGET_TOKENS


class PromptContextBudget(BaseModel):
    """Configuration for the retrieved-context budget.

    The budget is expressed in tokens (the natural unit of LLM context
    windows) and translated to an approximate character budget using a
    documented heuristic: ``chars_per_token`` characters per token. This
    avoids adding a tokenizer dependency while keeping the builder
    deterministic.

    The budget governs the RETRIEVED CONTEXT section only; the fixed system /
    grounding instructions and the user question are fixed overhead and are
    not counted against it.

    Attributes:
        max_tokens: Maximum tokens of retrieved context to include.
        chars_per_token: Approximate characters-per-token heuristic.
    """

    max_tokens: int = Field(
        default_factory=_default_budget_tokens,
        description="Maximum tokens allowed for retrieved context",
        gt=0,
    )
    chars_per_token: float = Field(
        default=DEFAULT_CHARS_PER_TOKEN,
        description="Approximate characters per token heuristic",
        gt=0.0,
    )

    @property
    def max_chars(self) -> int:
        """Approximate character budget derived from the token budget."""
        return int(self.max_tokens * self.chars_per_token)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "max_tokens": 2000,
                "chars_per_token": 4.0,
            }
        }
    )


class PromptContextItem(BaseModel):
    """A single retrieved result rendered into the prompt.

    This is a normalized, prompt-facing view of a RetrievalResult so the
    built prompt is fully self-describing and easy to inspect.

    Attributes:
        rank: 1-based relevance rank (most relevant first).
        file: Relative file path of the chunk.
        language: Programming language of the chunk.
        chunk_type: Type of chunk (function, class, module, ...).
        start_line: Starting line number of the chunk (1-based).
        end_line: Ending line number of the chunk (1-based).
        relevance_score: Normalized relevance in [0, 1]; higher = more relevant.
        content: The chunk content (source code / text).
    """

    rank: int = Field(..., description="1-based relevance rank", ge=1)
    file: str = Field(..., description="Relative file path of the chunk")
    language: str = Field(..., description="Programming language of the chunk")
    chunk_type: str = Field(..., description="Type of chunk")
    start_line: int = Field(..., description="Starting line number (1-based)", ge=1)
    end_line: int = Field(..., description="Ending line number (1-based)", ge=1)
    relevance_score: float = Field(
        ...,
        description="Normalized relevance in [0, 1]",
        ge=0.0,
        le=1.0,
    )
    content: str = Field(..., description="The chunk content")


class BuiltPrompt(BaseModel):
    """Result of building a prompt for the LLM.

    Contains the final prompt string plus deterministic bookkeeping about
    exactly what was included and what was dropped for budget reasons, which
    the RAG service and tests can inspect. The PromptBuilder NEVER calls an
    LLM; it only produces this model.

    Attributes:
        prompt: The final, inspectable prompt string.
        repository_name: Repository the question was asked about.
        question: The user question.
        total_results: Number of retrieved results passed to the builder.
        included_results: The results actually rendered into the prompt
            (complete chunks, in rank order).
        excluded_results: Results dropped because they did not fit the budget.
        total_chars: Total length of the prompt string.
        budget_max_tokens: Token budget configured.
        budget_max_chars: Approximate character budget derived from it.
        budget_used_chars: Characters of retrieved context actually included.
        budget_exceeded: True when not all retrieved results could fit.
    """

    prompt: str = Field(..., description="The final prompt string")
    repository_name: str = Field(..., description="Repository queried")
    question: str = Field(..., description="The user question")
    total_results: int = Field(..., description="Number of retrieved results", ge=0)
    included_results: List[PromptContextItem] = Field(
        default_factory=list,
        description="Results rendered into the prompt (complete chunks)",
    )
    excluded_results: int = Field(
        ...,
        description="Results dropped because they exceeded the budget",
        ge=0,
    )
    total_chars: int = Field(..., description="Total prompt length in characters", ge=0)
    budget_max_tokens: int = Field(..., description="Configured token budget", gt=0)
    budget_max_chars: int = Field(..., description="Approximate char budget", gt=0)
    budget_used_chars: int = Field(
        ...,
        description="Characters of retrieved context included",
        ge=0,
    )
    budget_exceeded: bool = Field(
        ...,
        description="True when not all results fit within the budget",
    )

    @property
    def included_count(self) -> int:
        """Number of context blocks rendered into the prompt."""
        return len(self.included_results)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prompt": "You are CodePilot...\nUSER QUESTION\n...",
                "repository_name": "demo-repo",
                "question": "Where is user authentication handled?",
                "total_results": 3,
                "included_results": [],
                "excluded_results": 1,
                "total_chars": 2048,
                "budget_max_tokens": 2000,
                "budget_max_chars": 8000,
                "budget_used_chars": 5120,
                "budget_exceeded": True,
            }
        }
    )


__all__ = [
    "BuiltPrompt",
    "PromptContextBudget",
    "PromptContextItem",
    "DEFAULT_CHARS_PER_TOKEN",
    "DEFAULT_CONTEXT_BUDGET_TOKENS",
]
