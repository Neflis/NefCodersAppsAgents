"""Message primitives for asynchronous event communication."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4


class EventType(StrEnum):
    """Known event types exchanged by agents."""

    GOAL_SUBMITTED = "GOAL_SUBMITTED"
    GOAL_DECOMPOSED = "GOAL_DECOMPOSED"
    TASK_CREATED = "TASK_CREATED"
    TASK_READY = "TASK_READY"
    TASK_CLAIMED = "TASK_CLAIMED"
    TASK_BLOCKED = "TASK_BLOCKED"
    TASK_RETRIED = "TASK_RETRIED"
    TASK_COMPLETED = "TASK_COMPLETED"
    CODE_PROPOSED = "CODE_PROPOSED"
    REVIEW_APPROVED = "REVIEW_APPROVED"
    REVIEW_REJECTED = "REVIEW_REJECTED"
    PROJECT_REVIEW_REQUESTED = "PROJECT_REVIEW_REQUESTED"
    PROJECT_REVIEW_APPROVED = "PROJECT_REVIEW_APPROVED"
    PROJECT_REVIEW_REJECTED = "PROJECT_REVIEW_REJECTED"
    FILE_READ_REQUEST = "FILE_READ_REQUEST"
    FILE_WRITE_REQUEST = "FILE_WRITE_REQUEST"
    FILE_WRITTEN = "FILE_WRITTEN"
    FILE_WRITE_FAILED = "FILE_WRITE_FAILED"
    TEST_PASSED = "TEST_PASSED"
    TEST_FAILED = "TEST_FAILED"
    TASK_FAILED = "TASK_FAILED"
    TASK_GRAPH_UPDATED = "TASK_GRAPH_UPDATED"
    WORKFLOW_STARTED = "WORKFLOW_STARTED"
    WORKFLOW_COMPLETED = "WORKFLOW_COMPLETED"
    WORKFLOW_HALTED = "WORKFLOW_HALTED"
    WORKFLOW_TIMEOUT = "WORKFLOW_TIMEOUT"


MessageType = EventType


@dataclass(slots=True)
class Message:
    """Event exchanged through the asynchronous message bus."""

    sender: str
    type: str | EventType
    content: Any
    receiver: str = "*"
    priority: int = 0
    correlation_id: str | None = None
    causation_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        """Default correlation to the event id for root events."""
        if self.correlation_id is None:
            self.correlation_id = self.id
