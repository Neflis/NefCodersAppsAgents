"""Local settings loaded from environment and optional .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_DATABASE_URL = "sqlite:///multi_agent_lab.db"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 10.0
DEFAULT_USE_MOCK_LLM = True


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
    ollama_model_planner: str = DEFAULT_OLLAMA_MODEL
    ollama_model_coder: str = DEFAULT_OLLAMA_MODEL
    ollama_model_reviewer: str = DEFAULT_OLLAMA_MODEL
    ollama_timeout_seconds: float = DEFAULT_OLLAMA_TIMEOUT_SECONDS
    use_mock_llm: bool = DEFAULT_USE_MOCK_LLM
    database_url: str = DEFAULT_DATABASE_URL


def load_settings(env_file: str | Path = ".env") -> Settings:
    """Load settings from environment variables and an optional .env file."""
    load_dotenv(env_file)
    return Settings(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
        ollama_model=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
        ollama_model_planner=os.getenv("OLLAMA_MODEL_PLANNER", DEFAULT_OLLAMA_MODEL),
        ollama_model_coder=os.getenv("OLLAMA_MODEL_CODER", DEFAULT_OLLAMA_MODEL),
        ollama_model_reviewer=os.getenv("OLLAMA_MODEL_REVIEWER", DEFAULT_OLLAMA_MODEL),
        ollama_timeout_seconds=float(
            os.getenv("OLLAMA_TIMEOUT_SECONDS", str(DEFAULT_OLLAMA_TIMEOUT_SECONDS))
        ),
        use_mock_llm=os.getenv("USE_MOCK_LLM", str(DEFAULT_USE_MOCK_LLM)).lower()
        in {"1", "true", "yes", "on"},
        database_url=os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL),
    )


settings = load_settings()
