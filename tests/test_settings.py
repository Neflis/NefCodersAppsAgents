from multi_agent_lab.config.settings import load_settings


def test_load_settings_reads_env_file(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "OLLAMA_BASE_URL=http://localhost:11435",
                "OLLAMA_MODEL=test-model",
                "DATABASE_URL=sqlite:///custom.db",
                "MAX_EVENTS_PER_WORKFLOW=300",
                "MAX_FIX_ATTEMPTS=4",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    settings = load_settings(env_file)

    assert settings.ollama_base_url == "http://localhost:11435"
    assert settings.ollama_model == "test-model"
    assert settings.database_url == "sqlite:///custom.db"
    assert settings.max_events_per_workflow == 300
    assert settings.max_fix_attempts == 4
