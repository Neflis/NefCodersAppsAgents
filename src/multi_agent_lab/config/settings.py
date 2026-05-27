"""Local settings for the multi-agent lab."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Settings:
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"


settings = Settings()
