"""Message primitives for asynchronous agent communication."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4


class MessageType(StrEnum):
    """Known message types exchanged by agents."""

    TASK_CREATED = "task.created"
    CODE_SIMULATED = "code.simulated"
    REVIEW_COMPLETED = "review.completed"
    FILE_READ_REQUEST = "FILE_READ_REQUEST"
    FILE_WRITE_REQUEST = "FILE_WRITE_REQUEST"
    FILE_WRITE_RESULT = "FILE_WRITE_RESULT"
    TASK_FAILED = "TASK_FAILED"


@dataclass(slots=True)
class Message:
    """Message exchanged through the asynchronous message bus."""

    sender: str
    receiver: str
    type: str | MessageType
    content: Any
    priority: int = 0
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
