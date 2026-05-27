"""Local settings loaded from environment and optional .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_DATABASE_URL = "sqlite:///multi_agent_lab.db"


def load_dotenv(path: str | Path = ".env") -> None:
    """Load key-value pairs from a local .env file."""
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime configuration values."""

    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    database_url: str = DEFAULT_DATABASE_URL


def load_settings(env_file: str | Path = ".env") -> Settings:
    """Load settings from environment variables and an optional .env file."""
    load_dotenv(env_file)
    return Settings(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
        ollama_model=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
        database_url=os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL),
    )


settings = load_settings()
