"""Agent event logging backed by the persistence store."""

from __future__ import annotations

from typing import Any

from multi_agent_lab.core.sqlite_store import SQLiteStore


class AgentEventLogger:
    """Records agent events through the configured store."""

    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    async def log(self, agent: str, event_type: str, details: dict[str, Any] | None = None) -> None:
        """Persist one agent event."""
        self.store.save_agent_event(agent, event_type, details)
