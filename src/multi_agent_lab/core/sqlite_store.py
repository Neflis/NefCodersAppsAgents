"""SQLite persistence for messages, tasks, and agent events."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from multi_agent_lab.core.message import Message
from multi_agent_lab.core.task import Task, TaskStatus


def database_path_from_url(database_url: str) -> str:
    if database_url == ":memory:":
        return database_url

    parsed = urlparse(database_url)
    if parsed.scheme == "sqlite":
        if parsed.netloc and parsed.path:
            return f"//{parsed.netloc}{parsed.path}"
        return parsed.path.lstrip("/") or ":memory:"

    return database_url


class SQLiteStore:
    def __init__(self, database_url: str = "sqlite:///multi_agent_lab.db") -> None:
        self.database_url = database_url
        self.database_path = database_path_from_url(database_url)
        if self.database_path != ":memory:":
            Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.database_path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self.initialize()

    def initialize(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                sender TEXT NOT NULL,
                receiver TEXT NOT NULL,
                type TEXT NOT NULL,
                content TEXT NOT NULL,
                priority INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                payload TEXT NOT NULL,
                priority INTEGER NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS agent_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent TEXT NOT NULL,
                event_type TEXT NOT NULL,
                details TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        self._connection.commit()

    def save_message(self, message: Message) -> None:
        self._connection.execute(
            """
            INSERT OR REPLACE INTO messages
            (id, sender, receiver, type, content, priority, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message.id,
                message.sender,
                message.receiver,
                message.type,
                json.dumps(message.content, ensure_ascii=True),
                message.priority,
                message.created_at.isoformat(),
            ),
        )
        self._connection.commit()

    def save_task(self, task: Task) -> None:
        self._connection.execute(
            """
            INSERT OR REPLACE INTO tasks
            (id, title, description, payload, priority, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                task.id,
                task.title,
                task.description,
                json.dumps(task.payload, ensure_ascii=True),
                task.priority,
                task.status.value,
                task.created_at.isoformat(),
            ),
        )
        self._connection.commit()

    def update_task_status(self, task_id: str, status: TaskStatus) -> None:
        self._connection.execute(
            "UPDATE tasks SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (status.value, task_id),
        )
        self._connection.commit()

    def save_agent_event(self, agent: str, event_type: str, details: dict[str, Any] | None = None) -> None:
        self._connection.execute(
            """
            INSERT INTO agent_events (agent, event_type, details, created_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (agent, event_type, json.dumps(details or {}, ensure_ascii=True)),
        )
        self._connection.commit()

    def fetch_all(self, table: str) -> list[sqlite3.Row]:
        if table not in {"messages", "tasks", "agent_events"}:
            raise ValueError(f"Unsupported table: {table}")
        cursor = self._connection.execute(f"SELECT * FROM {table}")
        return list(cursor.fetchall())

    def close(self) -> None:
        self._connection.close()
