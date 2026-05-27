"""Small async Ollama client using the Python standard library."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OllamaClientError(RuntimeError):
    """Raised when Ollama cannot complete a request."""


@dataclass(slots=True)
class OllamaClient:
    """Minimal client for the local Ollama HTTP API."""

    base_url: str = "http://localhost:11434"
    model: str = "llama3.2"
    timeout: float = 10.0

    async def health_check(self) -> bool:
        """Return whether the Ollama API is reachable."""
        try:
            await asyncio.to_thread(self._request, "GET", "/api/tags")
        except OllamaClientError:
            return False
        return True

    async def generate(self, prompt: str) -> str:
        """Generate text with the configured Ollama model."""
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        data = await asyncio.to_thread(self._request, "POST", "/api/generate", payload)
        response = data.get("response")
        if not isinstance(response, str):
            raise OllamaClientError("Ollama response did not include text.")
        return response

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Execute one JSON HTTP request against Ollama."""
        url = f"{self.base_url.rstrip('/')}{path}"
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            url,
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as error:
            raise OllamaClientError(f"Ollama HTTP error {error.code}.") from error
        except URLError as error:
            raise OllamaClientError(f"Ollama connection error: {error.reason}") from error
        except TimeoutError as error:
            raise OllamaClientError("Ollama request timed out.") from error

        try:
            return json.loads(raw or "{}")
        except json.JSONDecodeError as error:
            raise OllamaClientError("Ollama returned invalid JSON.") from error
