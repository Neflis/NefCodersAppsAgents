"""SQLite persistence for messages, tasks, and agent events."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from multi_agent_lab.core.message import Message
from multi_agent_lab.core.task import Task, TaskStatus
from multi_agent_lab.core.task_graph import TaskGraph


def database_path_from_url(database_url: str) -> str:
    """Convert a SQLite URL into a filesystem path."""
    if database_url == ":memory:":
        return database_url

    parsed = urlparse(database_url)
    if parsed.scheme == "sqlite":
        if parsed.netloc and parsed.path:
            return f"//{parsed.netloc}{parsed.path}"
        return parsed.path.lstrip("/") or ":memory:"

    return database_url


class SQLiteStore:
    """Small SQLite-backed persistence store."""

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
        """Create persistence tables if they do not already exist."""
        self._connection.executescript("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                sender TEXT NOT NULL,
                receiver TEXT NOT NULL,
                type TEXT NOT NULL,
                content TEXT NOT NULL,
                priority INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                correlation_id TEXT,
                causation_id TEXT,
                metadata TEXT NOT NULL DEFAULT '{}'
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

            CREATE TABLE IF NOT EXISTS task_graphs (
                correlation_id TEXT PRIMARY KEY,
                goal_id TEXT NOT NULL,
                graph TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS project_memory (
                correlation_id TEXT PRIMARY KEY,
                memory TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """)
        self._migrate_messages_table()
        self._connection.commit()

    def _migrate_messages_table(self) -> None:
        """Add newer message columns to existing databases."""
        existing_columns = {
            row["name"] for row in self._connection.execute("PRAGMA table_info(messages)")
        }
        migrations = {
            "correlation_id": "ALTER TABLE messages ADD COLUMN correlation_id TEXT",
            "causation_id": "ALTER TABLE messages ADD COLUMN causation_id TEXT",
            "metadata": "ALTER TABLE messages ADD COLUMN metadata TEXT NOT NULL DEFAULT '{}'",
        }
        for column, statement in migrations.items():
            if column not in existing_columns:
                self._connection.execute(statement)

    def save_message(self, message: Message) -> None:
        """Persist a message."""
        self._connection.execute(
            """
            INSERT OR REPLACE INTO messages
            (
                id,
                sender,
                receiver,
                type,
                content,
                priority,
                created_at,
                correlation_id,
                causation_id,
                metadata
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message.id,
                message.sender,
                message.receiver,
                str(message.type),
                json.dumps(message.content, ensure_ascii=True),
                message.priority,
                message.created_at.isoformat(),
                message.correlation_id,
                message.causation_id,
                json.dumps(message.metadata, ensure_ascii=True),
            ),
        )
        self._connection.commit()

    def save_task(self, task: Task) -> None:
        """Persist a task."""
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
        """Persist a task status change."""
        self._connection.execute(
            "UPDATE tasks SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (status.value, task_id),
        )
        self._connection.commit()

    def save_agent_event(
        self, agent: str, event_type: str, details: dict[str, Any] | None = None
    ) -> None:
        """Persist an agent lifecycle or processing event."""
        self._connection.execute(
            """
            INSERT INTO agent_events (agent, event_type, details, created_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (agent, event_type, json.dumps(details or {}, ensure_ascii=True)),
        )
        self._connection.commit()

    def save_task_graph(self, graph: TaskGraph) -> None:
        """Persist a task graph snapshot."""
        self._connection.execute(
            """
            INSERT OR REPLACE INTO task_graphs (correlation_id, goal_id, graph, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                graph.goal.correlation_id,
                graph.goal.id,
                json.dumps(graph.to_dict(), ensure_ascii=True),
            ),
        )
        self._connection.commit()

    def save_project_memory(self, memory: dict[str, Any]) -> None:
        """Persist semantic project memory."""
        self._connection.execute(
            """
            INSERT OR REPLACE INTO project_memory (correlation_id, memory, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            """,
            (
                str(memory["correlation_id"]),
                json.dumps(memory, ensure_ascii=True),
            ),
        )
        self._connection.commit()

    def load_project_memory(self, correlation_id: str) -> dict[str, Any] | None:
        """Load semantic project memory by correlation id."""
        cursor = self._connection.execute(
            "SELECT memory FROM project_memory WHERE correlation_id = ?",
            (correlation_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return dict(json.loads(str(row["memory"])))

    def fetch_all(self, table: str) -> list[sqlite3.Row]:
        """Fetch all rows from one supported table."""
        if table not in {
            "messages",
            "tasks",
            "agent_events",
            "task_graphs",
            "project_memory",
        }:
            raise ValueError(f"Unsupported table: {table}")
        cursor = self._connection.execute(f"SELECT * FROM {table}")
        return list(cursor.fetchall())

    def close(self) -> None:
        """Close the database connection."""
        self._connection.close()
