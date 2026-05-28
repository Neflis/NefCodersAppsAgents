"""Async in-memory event bus."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Iterable

from multi_agent_lab.core.event_noise import EventNoiseReducer
from multi_agent_lab.core.message import EventType, Message
from multi_agent_lab.core.project_memory_service import ProjectMemoryService
from multi_agent_lab.core.sqlite_store import SQLiteStore

WILDCARD_EVENT = "*"


class MessageBus:
    """In-memory async pub/sub bus with optional persistence."""

    def __init__(
        self,
        store: SQLiteStore | None = None,
        project_memory: ProjectMemoryService | None = None,
        noise_reducer: EventNoiseReducer | None = None,
    ) -> None:
        self._subscribers: dict[str, list[asyncio.Queue[Message]]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._store = store
        self._project_memory = project_memory
        self._noise_reducer = noise_reducer

    async def subscribe(
        self,
        event_type: str | EventType,
        queue: asyncio.Queue[Message] | None = None,
    ) -> asyncio.Queue[Message]:
        """Subscribe a queue to one event type."""
        inbox: asyncio.Queue[Message] = queue or asyncio.Queue()
        async with self._lock:
            self._subscribers[str(event_type)].append(inbox)
        return inbox

    async def subscribe_many(
        self,
        event_types: Iterable[str | EventType],
    ) -> asyncio.Queue[Message]:
        """Subscribe one queue to multiple event types."""
        inbox: asyncio.Queue[Message] = asyncio.Queue()
        for event_type in event_types:
            await self.subscribe(event_type, inbox)
        return inbox

    async def publish(self, message: Message) -> None:
        """Publish an event to matching and wildcard subscribers."""
        if self._store is not None:
            self._store.save_message(message)
        if self._project_memory is not None:
            self._project_memory.update_from_event(message)
        if self._noise_reducer is not None:
            self._noise_reducer.record(message)

        async with self._lock:
            event_subscribers = list(self._subscribers.get(str(message.type), []))
            wildcard_subscribers = list(self._subscribers.get(WILDCARD_EVENT, []))

        delivered: set[int] = set()
        for queue in event_subscribers + wildcard_subscribers:
            queue_id = id(queue)
            if queue_id in delivered:
                continue
            delivered.add(queue_id)
            await queue.put(message)
