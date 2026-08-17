"""Prompt building and management.

Sprint 8 adds the PromptBuilder, which converts a user question, a repository
name, and retrieved RetrievalResult objects into a deterministic, grounded
LLM prompt. The builder NEVER calls an LLM; it only produces a BuiltPrompt.

The PromptBuilder enforces a configurable context budget so that only
complete retrieved chunks that fit within the budget are included, using a
documented character-based approximation of token counts (no tokenizer
dependency).

Example usage:
    from app.prompts import PromptBuilder

    builder = PromptBuilder()
    built = builder.build(
        question="Where is user authentication handled?",
        repository_name="demo-repo",
        results=retrieved_results,
    )
    print(built.prompt)
    print(built.budget_exceeded, built.excluded_results)
"""

import logging

from app.prompts.builder import PromptBuilder
from app.prompts.models import (
    DEFAULT_CHARS_PER_TOKEN,
    DEFAULT_CONTEXT_BUDGET_TOKENS,
    BuiltPrompt,
    PromptContextBudget,
    PromptContextItem,
)

logger = logging.getLogger(__name__)


def create_prompt_builder(budget=None) -> PromptBuilder:
    """Create a configured PromptBuilder instance.

    Convenience factory for wiring a PromptBuilder with an optional context
    budget.

    Args:
        budget: A PromptContextBudget. If None, a default budget is created
            from configuration (LLM_CONTEXT_BUDGET_TOKENS env).

    Returns:
        A configured PromptBuilder instance
    """
    builder = PromptBuilder(budget=budget)
    logger.info(f"Created prompt builder: {builder.__class__.__name__}")
    return builder


__all__ = [
    "PromptBuilder",
    "PromptContextBudget",
    "PromptContextItem",
    "BuiltPrompt",
    "DEFAULT_CHARS_PER_TOKEN",
    "DEFAULT_CONTEXT_BUDGET_TOKENS",
    "create_prompt_builder",
]