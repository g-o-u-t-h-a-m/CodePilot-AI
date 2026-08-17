"""Data models for the LLM generation layer.

These models are application-facing and independent of any specific LLM
provider. LLMResponse is the normalized result of a generation call,
produced by any LLMProvider implementation.
"""

from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class LLMResponse(BaseModel):
    """A normalized response from an LLM provider.

    Every provider (mock, OpenAI-compatible HTTP, ...) returns this same
    shape so that the RAG service never depends on a concrete provider.

    Attributes:
        content: The generated text.
        model_name: The model that produced the content, if known.
        provider_name: Name of the provider that produced the response.
        metadata: Provider-specific information (e.g. usage, latency).
    """

    content: str = Field(
        ...,
        description="The generated text content",
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Name of the model used for generation",
    )
    provider_name: str = Field(
        ...,
        description="Name of the provider that produced the response",
    )
    metadata: Dict[str, object] = Field(
        default_factory=dict,
        description="Provider-specific metadata (usage, latency, ...)",
    )

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "content": "User authentication is handled in src/auth.py (lines 10-25).",
                "model_name": "mock-llm",
                "provider_name": "mock",
                "metadata": {},
            }
        }
    )


__all__ = ["LLMResponse"]