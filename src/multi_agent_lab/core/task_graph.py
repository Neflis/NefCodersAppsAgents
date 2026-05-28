"""Dynamic task graph for goal decomposition and coordination."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4


class TaskNodeStatus(StrEnum):
    """Task graph node lifecycle states."""

    PENDING = "PENDING"
    READY = "READY"
    IN_PROGRESS = "IN_PROGRESS"
    BLOCKED = "BLOCKED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass(slots=True)
class Goal:
    """Top-level goal tracked by a task graph."""

    title: str
    correlation_id: str
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the goal."""
        return {
            "id": self.id,
            "title": self.title,
            "correlation_id": self.correlation_id,
            "created_at": self.created_at.isoformat(),
        }


@dataclass(slots=True)
class TaskNode:
    """One task node in a dynamic task graph."""

    title: str
    description: str
    required_capability: str
    payload: dict[str, Any] = field(default_factory=dict)
    dependencies: set[str] = field(default_factory=set)
    subtasks: list[str] = field(default_factory=list)
    priority: int = 0
    status: TaskNodeStatus = TaskNodeStatus.PENDING
    owner: str | None = None
    retries: int = 0
    max_retries: int = 2
    result: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def mark(self, status: TaskNodeStatus, owner: str | None = None) -> None:
        """Update status, owner, and timestamp."""
        self.status = status
        if owner is not None:
            self.owner = owner
        self.updated_at = datetime.now(UTC)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the task node."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "required_capability": self.required_capability,
            "payload": self.payload,
            "dependencies": sorted(self.dependencies),
            "subtasks": self.subtasks,
            "priority": self.priority,
            "status": self.status.value,
            "owner": self.owner,
            "retries": self.retries,
            "max_retries": self.max_retries,
            "result": self.result,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class TaskGraph:
    """In-memory task graph for one goal."""

    def __init__(self, goal: Goal) -> None:
        self.goal = goal
        self.nodes: dict[str, TaskNode] = {}
        self.created_at = datetime.now(UTC)
        self.updated_at = self.created_at

    def add_task(self, node: TaskNode) -> TaskNode:
        """Add one task node to the graph."""
        self.nodes[node.id] = node
        self.updated_at = datetime.now(UTC)
        return node

    def add_dependency(self, task_id: str, dependency_id: str) -> None:
        """Make one task depend on another."""
        self.nodes[task_id].dependencies.add(dependency_id)
        self.updated_at = datetime.now(UTC)

    def ready_tasks(self) -> list[TaskNode]:
        """Return pending tasks whose dependencies are completed."""
        ready: list[TaskNode] = []
        for node in self.nodes.values():
            if node.status not in {TaskNodeStatus.PENDING, TaskNodeStatus.BLOCKED}:
                continue
            if self.dependencies_completed(node.id):
                node.mark(TaskNodeStatus.READY)
                ready.append(node)
        self.updated_at = datetime.now(UTC)
        return sorted(ready, key=lambda task: task.priority, reverse=True)

    def dependencies_completed(self, task_id: str) -> bool:
        """Return whether all dependencies for a task are completed."""
        node = self.nodes[task_id]
        return all(
            self.nodes[dependency_id].status == TaskNodeStatus.COMPLETED
            for dependency_id in node.dependencies
        )

    def claim_task(self, task_id: str, owner: str) -> TaskNode:
        """Mark a ready task as in progress."""
        node = self.nodes[task_id]
        node.mark(TaskNodeStatus.IN_PROGRESS, owner)
        self.updated_at = datetime.now(UTC)
        return node

    def complete_task(self, task_id: str, result: dict[str, Any]) -> TaskNode:
        """Mark a task completed and store its result."""
        node = self.nodes[task_id]
        node.result = result
        node.mark(TaskNodeStatus.COMPLETED)
        self.updated_at = datetime.now(UTC)
        return node

    def fail_task(self, task_id: str, error: str) -> TaskNode:
        """Mark a task failed and store its error."""
        node = self.nodes[task_id]
        node.result = {"error": error}
        node.mark(TaskNodeStatus.FAILED)
        self.updated_at = datetime.now(UTC)
        return node

    def retry_task(self, task_id: str) -> TaskNode:
        """Move a failed task back to ready if retry budget remains."""
        node = self.nodes[task_id]
        node.retries += 1
        node.mark(TaskNodeStatus.READY)
        self.updated_at = datetime.now(UTC)
        return node

    def block_task(self, task_id: str, reason: str) -> TaskNode:
        """Block a task after retries are exhausted."""
        node = self.nodes[task_id]
        node.result = {"error": reason}
        node.mark(TaskNodeStatus.BLOCKED)
        self.updated_at = datetime.now(UTC)
        return node

    def dependency_results(self, task_id: str) -> list[dict[str, Any]]:
        """Return completed dependency result payloads."""
        return [
            self.nodes[dependency_id].result for dependency_id in self.nodes[task_id].dependencies
        ]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the graph."""
        return {
            "goal": self.goal.to_dict(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "nodes": {task_id: node.to_dict() for task_id, node in self.nodes.items()},
        }
