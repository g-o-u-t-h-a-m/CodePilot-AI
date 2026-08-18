"""Minimal HTTP client for the CodePilot backend.

``CodePilotClient`` is the single abstraction the CLI uses to talk to the
FastAPI backend. It constructs requests, encodes/decodes JSON, and translates
network/HTTP failures into typed exceptions the command layer can render as
human-readable messages. It contains NO business logic: cloning, scanning,
chunking, embedding, retrieval, and generation all remain inside the backend
(``app/api`` -> ``app/services``). The client only marshals the existing API
contracts:

    GET  /health
    POST /repository/clone
    POST /repository/index
    POST /rag/query

Only the Python standard library (``urllib``) is used, so the client runs in
any environment without extra dependencies and works on Windows/PowerShell.
"""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from typing import Dict, Optional

#: Default backend URL when nothing else is configured.
DEFAULT_BASE_URL = "http://127.0.0.1:8000"

#: Environment variable that overrides the backend URL.
API_URL_ENV = "CODEPILOT_API_URL"

#: Default per-request timeout in seconds.
DEFAULT_TIMEOUT = 30.0


class CodePilotError(Exception):
    """Base class for all errors raised by the CodePilot client."""


class CodePilotConnectionError(CodePilotError):
    """The API could not be reached (connection refused, DNS, timeout).

    Attributes:
        reason: Underlying failure description, if any.
    """

    def __init__(self, reason: str = ""):
        self.reason = reason
        super().__init__(reason or "connection failed")


class CodePilotHTTPError(CodePilotError):
    """The API returned a non-2xx HTTP status.

    Attributes:
        status_code: HTTP status code.
        detail: Human-readable detail parsed from the error body.
        body: Raw response body (kept for debugging, never shown raw).
    """

    def __init__(self, status_code: int, detail: str = "", body: str = ""):
        self.status_code = status_code
        self.detail = detail or f"HTTP {status_code}"
        self.body = body
        super().__init__(f"HTTP {status_code}: {self.detail}")


class CodePilotProtocolError(CodePilotError):
    """The API response was not valid JSON or had an unexpected shape."""


class CodePilotClient:
    """HTTP client for the CodePilot FastAPI backend.

    Args:
        base_url: Backend base URL. When ``None``, falls back to the
            ``CODEPILOT_API_URL`` environment variable, then the default
            ``http://127.0.0.1:8000``.
        timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        """Initialize the client with a resolved base URL."""
        configured = base_url or os.environ.get(API_URL_ENV) or DEFAULT_BASE_URL
        self.base_url = str(configured).rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public API: one method per backend endpoint
    # ------------------------------------------------------------------

    def health(self) -> Dict:
        """GET /health.

        Returns:
            The health payload, e.g. ``{"status": "healthy", ...}``.
        """
        return self._request("GET", "/health")

    def clone_repository(self, github_url: str) -> Dict:
        """POST /repository/clone.

        Args:
            github_url: GitHub repository URL.

        Returns:
            The ``CloneRepositoryResponse`` payload.
        """
        return self._request(
            "POST",
            "/repository/clone",
            payload={"github_url": github_url},
        )

    def index_repository(self, repository_name: str) -> Dict:
        """POST /repository/index.

        Args:
            repository_name: Name of an already-cloned repository.

        Returns:
            The ``IndexRepositoryResponse`` payload.
        """
        return self._request(
            "POST",
            "/repository/index",
            payload={"repository_name": repository_name},
        )

    def query(
        self,
        repository_name: str,
        question: str,
        top_k: Optional[int] = None,
    ) -> Dict:
        """POST /rag/query.

        Args:
            repository_name: Name of an indexed repository.
            question: Natural-language question.
            top_k: Optional maximum number of sources to retrieve.

        Returns:
            The ``RagQueryResponse`` payload.
        """
        payload: Dict = {"repository_name": repository_name, "question": question}
        if top_k is not None:
            payload["top_k"] = top_k
        return self._request("POST", "/rag/query", payload=payload)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _request(self, method: str, path: str, payload: Optional[Dict] = None) -> Dict:
        """Perform a JSON HTTP request against the backend.

        Args:
            method: HTTP method (GET/POST).
            path: Absolute URL path, e.g. ``/health``.
            payload: Optional JSON body.

        Returns:
            The decoded JSON response object.

        Raises:
            CodePilotConnectionError: If the API is unreachable.
            CodePilotHTTPError: If the API returns a non-2xx status.
            CodePilotProtocolError: If the response is malformed JSON.
        """
        url = self.base_url + path
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(url, data=data, headers=headers, method=method)

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            raise CodePilotHTTPError(
                status_code=exc.code,
                detail=self._extract_detail(raw),
                body=raw,
            ) from exc
        except (urllib.error.URLError, socket.timeout, ConnectionError, OSError) as exc:
            reason = self._describe_failure(exc)
            raise CodePilotConnectionError(reason) from exc

        try:
            if not raw.strip():
                return {}
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CodePilotProtocolError(
                f"Malformed JSON response from the API: {exc}"
            ) from exc

        if not isinstance(decoded, dict):
            raise CodePilotProtocolError(
                "Unexpected response shape from the API: expected a JSON "
                f"object, got {type(decoded).__name__}"
            )
        return decoded

    @staticmethod
    def _extract_detail(raw: str) -> str:
        """Extract a human-readable detail string from an error body.

        FastAPI returns ``{"detail": "<message>"}`` for HTTPExceptions and a
        list of per-field messages for 422 validation errors.

        Args:
            raw: Raw response text.

        Returns:
            A single-line detail message.
        """
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            return raw.strip() or "Unknown error"

        if isinstance(body, dict):
            detail = body.get("detail")
            if isinstance(detail, str):
                return detail.strip() or "Unknown error"
            if isinstance(detail, list):
                messages = []
                for item in detail:
                    if isinstance(item, dict):
                        location = ".".join(
                            str(part) for part in item.get("loc", []) if part != "body"
                        )
                        msg = str(item.get("msg", "")).strip()
                        if location and msg:
                            messages.append(f"{location}: {msg}")
                        elif msg:
                            messages.append(msg)
                if messages:
                    return "; ".join(messages)
        elif isinstance(body, str):
            return body.strip()

        return raw.strip() or "Unknown error"

    @staticmethod
    def _describe_failure(exc: Exception) -> str:
        """Describe a connection-level failure without a traceback.

        Args:
            exc: The low-level exception raised by urllib.

        Returns:
            A short description of the failure reason.
        """
        reason = getattr(exc, "reason", None)
        if reason is not None:
            return str(reason)
        return str(exc) or exc.__class__.__name__


__all__ = [
    "CodePilotClient",
    "CodePilotError",
    "CodePilotConnectionError",
    "CodePilotHTTPError",
    "CodePilotProtocolError",
    "API_URL_ENV",
    "DEFAULT_BASE_URL",
]
