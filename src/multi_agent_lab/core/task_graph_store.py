"""In-memory registry for task graphs."""

from __future__ import annotations

from multi_agent_lab.core.sqlite_store import SQLiteStore
from multi_agent_lab.core.task_graph import TaskGraph


class TaskGraphStore:
    """Stores task graphs by correlation id and persists snapshots."""

    def __init__(self, store: SQLiteStore | None = None) -> None:
        self._graphs: dict[str, TaskGraph] = {}
        self._store = store

    def add(self, graph: TaskGraph) -> None:
        """Add or replace a task graph."""
        self._graphs[graph.goal.correlation_id] = graph
        self.persist(graph)

    def get(self, correlation_id: str) -> TaskGraph:
        """Return a graph by correlation id."""
        return self._graphs[correlation_id]

    def persist(self, graph: TaskGraph) -> None:
        """Persist a graph snapshot if persistence is configured."""
        if self._store is not None:
            self._store.save_task_graph(graph)
