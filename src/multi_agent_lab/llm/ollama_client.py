"""Placeholder client for a future Ollama integration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class OllamaClient:
    base_url: str = "http://localhost:11434"
    model: str = "llama3.2"

    async def generate(self, prompt: str) -> str:
        raise NotImplementedError("Ollama integration will be implemented in a later phase.")
