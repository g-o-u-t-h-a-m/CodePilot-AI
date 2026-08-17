"""OpenAI-compatible HTTP LLM provider.

This provider talks to any OpenAI-compatible Chat Completions API over HTTP
using only the Python standard library (urllib), so it adds NO new
dependency to the project. It is the development-time bridge to services such
as OpenRouter that expose an OpenAI-compatible endpoint while keeping the
backend architecture provider-independent.

Configuration (environment variables, never hard-coded):
    LLM_PROVIDER=openai_compatible      (selects this provider)
    LLM_BASE_URL=http://.../v1          (Chat Completions endpoint)
    LLM_API_KEY=...                     (secret; do not commit)
    LLM_MODEL=...                       (model name; no specific model assumed)

Security: the API key is read from the environment only and is never logged
or embedded in code. If no API key is configured, generate() fails with a
clear message rather than attempting an unauthenticated call; the project's
0-budget mode relies on the MockLLMProvider instead.
"""

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Dict, Optional

from app.llm.models import LLMResponse
from app.llm.provider import LLMProvider

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(LLMProvider):
    """LLM provider for any OpenAI-compatible Chat Completions API.

    Attributes:
        base_url: Base URL of the API (e.g. https://openrouter.ai/api/v1).
        api_key: API key read from the LLM_API_KEY environment variable.
        model_name: Model identifier read from the LLM_MODEL environment
            variable. No specific model is assumed.
        timeout_seconds: Request timeout in seconds.
    """

    PROVIDER_NAME = "openai_compatible"

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout_seconds: int = 60,
    ):
        """Initialize the OpenAI-compatible provider.

        Args:
            base_url: API base URL. Defaults to LLM_BASE_URL, then to the
                OpenRouter default. The default is a documented convenience;
                secrets are never given defaults.
            api_key: API key. Defaults to LLM_API_KEY.
            model_name: Model identifier. Defaults to LLM_MODEL. If neither
                is set, the provider is still usable (the model is resolved
                lazily) but generates fail with a clear message.
            timeout_seconds: Request timeout.

        Raises:
            ValueError: If timeout_seconds is not positive.
        """
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}")

        self.base_url = (base_url or os.getenv("LLM_BASE_URL")
                         or "https://openrouter.ai/api/v1").rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("LLM_API_KEY")
        self.model_name = (model_name if model_name is not None
                           else os.getenv("LLM_MODEL", "unspecified"))
        self.timeout_seconds = timeout_seconds

        has_key = bool(self.api_key)
        logger.info(
            f"Initialized OpenAICompatibleProvider "
            f"(base_url: {self.base_url}, model: {self.model_name}, "
            f"api_key_configured: {has_key})"
        )
        if not has_key:
            logger.warning(
                "No LLM_API_KEY configured; generate() will fail until one "
                "is set. Use the MockLLMProvider for 0-cost local testing."
            )

    def generate(self, prompt: str) -> LLMResponse:
        """Generate a response from an OpenAI-compatible Chat Completions API.

        Args:
            prompt: The prompt produced by the PromptBuilder. Sent as a
                single user message with a system message describing the
                role, so grounding instructions already embedded in the
                prompt text are preserved verbatim.

        Returns:
            LLMResponse containing the generated content.

        Raises:
            RuntimeError: If no API key/model is configured, or the HTTP
                request fails (network, auth, non-2xx response, bad JSON).
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        if not self.api_key:
            raise RuntimeError(
                "LLM_API_KEY is not configured for provider "
                f"'{self.PROVIDER_NAME}'. Set LLM_API_KEY (and optionally "
                "LLM_MODEL/LLM_BASE_URL) or use the mock provider for "
                "0-cost local testing."
            )
        if self.model_name == "unspecified":
            raise RuntimeError(
                "LLM_MODEL is not configured for provider "
                f"'{self.PROVIDER_NAME}'. Set LLM_MODEL to the model you "
                "want to use (no specific model is assumed)."
            )

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are CodePilot, an AI assistant that analyzes "
                        "repository code. Follow the grounding rules in the "
                        "user message strictly."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
        }

        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        logger.info(
            f"Calling OpenAI-compatible API at {self.base_url}/chat/completions "
            f"(model: {self.model_name}, prompt chars: {len(prompt)})"
        )

        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:500]
            logger.error(
                f"LLM API returned HTTP {e.code} for provider "
                f"'{self.PROVIDER_NAME}': {detail}"
            )
            raise RuntimeError(
                f"LLM API HTTP {e.code} error: {detail}"
            ) from e
        except urllib.error.URLError as e:
            logger.error(
                f"LLM API network error for provider '{self.PROVIDER_NAME}': {e}"
            )
            raise RuntimeError(f"LLM API network error: {e}") from e
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error(
                f"LLM API returned invalid JSON for provider "
                f"'{self.PROVIDER_NAME}': {e}"
            )
            raise RuntimeError(f"LLM API returned invalid JSON: {e}") from e

        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Unexpected LLM API response shape: {body}")
            raise RuntimeError(
                f"Unexpected LLM API response shape: {e}"
            ) from e

        metadata: Dict[str, object] = {}
        usage = body.get("usage")
        if isinstance(usage, dict):
            metadata["usage"] = usage

        response = LLMResponse(
            content=content,
            model_name=self.model_name,
            provider_name=self.PROVIDER_NAME,
            metadata=metadata,
        )
        logger.info(
            f"LLM API returned {len(content)} chars (model: {self.model_name})"
        )
        return response

    def get_provider_name(self) -> str:
        """Get the provider name ('openai_compatible')."""
        return self.PROVIDER_NAME

    def get_model_name(self) -> str:
        """Get the configured model name."""
        return self.model_name


__all__ = ["OpenAICompatibleProvider"]