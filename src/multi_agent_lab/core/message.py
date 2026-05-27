"""Message primitives for asynchronous agent communication."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class Message:
    sender: str
    receiver: str
    type: str
    content: Any
    priority: int = 0
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
