"""Task model used by agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4


class TaskStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"


@dataclass(slots=True)
class Task:
    title: str
    description: str
    payload: dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    status: TaskStatus = TaskStatus.PENDING
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
