# Agent Instructions

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## Checks

```powershell
black .
ruff check .
pytest
python -m multi_agent_lab.main --mode demo_mock
```

## Style

- Python 3.11+.
- Keep modules small and explicit.
- Use event-driven communication through `MessageBus`.
- Prefer structured JSON decisions for LLM output.
- Keep tests focused on behavior and safety.

## Restrictions

- Do not use LangChain or CrewAI.
- Do not use `subprocess`.
- Do not add Docker.
- Do not add automatic git operations from agents.
- Do not execute shell commands from agents.
- Only `FileAgent` may write files, and only through `FileTool`.
- File operations must remain restricted to `./workspace/`.
