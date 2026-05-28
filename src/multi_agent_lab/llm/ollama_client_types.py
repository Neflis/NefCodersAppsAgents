"""Shared Ollama client exceptions."""


class OllamaClientError(RuntimeError):
    """Raised when Ollama cannot complete a request."""


class InvalidJSONError(OllamaClientError):
    """Raised when a model returns invalid JSON."""
