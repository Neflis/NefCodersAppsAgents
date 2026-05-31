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
    SPEC_REQUESTED = "SPEC_REQUESTED"
    SPEC_GENERATED = "SPEC_GENERATED"
    SPEC_APPROVED = "SPEC_APPROVED"
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
    PATCH_PROPOSED = "PATCH_PROPOSED"
    PATCH_APPLIED = "PATCH_APPLIED"
    PATCH_FAILED = "PATCH_FAILED"
    TEST_PASSED = "TEST_PASSED"
    TEST_FAILED = "TEST_FAILED"
    TEST_EXECUTION_REQUESTED = "TEST_EXECUTION_REQUESTED"
    TEST_EXECUTION_STARTED = "TEST_EXECUTION_STARTED"
    TEST_EXECUTION_PASSED = "TEST_EXECUTION_PASSED"
    TEST_EXECUTION_FAILED = "TEST_EXECUTION_FAILED"
    BUILD_STARTED = "BUILD_STARTED"
    BUILD_PASSED = "BUILD_PASSED"
    BUILD_FAILED = "BUILD_FAILED"
    FIX_REQUESTED = "FIX_REQUESTED"
    FIX_PROPOSED = "FIX_PROPOSED"
    FIX_APPLIED = "FIX_APPLIED"
    FIX_FAILED = "FIX_FAILED"
    RETEST_REQUESTED = "RETEST_REQUESTED"
    DEPENDENCY_INSTALL_REQUESTED = "DEPENDENCY_INSTALL_REQUESTED"
    DEPENDENCY_INSTALL_STARTED = "DEPENDENCY_INSTALL_STARTED"
    DEPENDENCY_INSTALL_SUCCEEDED = "DEPENDENCY_INSTALL_SUCCEEDED"
    DEPENDENCY_INSTALL_FAILED = "DEPENDENCY_INSTALL_FAILED"
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
