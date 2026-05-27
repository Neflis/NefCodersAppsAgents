"""Async in-memory message bus."""

from __future__ import annotations

import asyncio
from collections import defaultdict

from multi_agent_lab.core.message import Message
from multi_agent_lab.core.sqlite_store import SQLiteStore


class MessageBus:
    """In-memory async pub/sub bus with optional persistence."""

    def __init__(self, store: SQLiteStore | None = None) -> None:
        self._subscribers: dict[str, list[asyncio.Queue[Message]]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._store = store

    async def subscribe(self, receiver: str) -> asyncio.Queue[Message]:
        """Subscribe to messages addressed to a receiver."""
        queue: asyncio.Queue[Message] = asyncio.Queue()
        async with self._lock:
            self._subscribers[receiver].append(queue)
        return queue

    async def publish(self, message: Message) -> None:
        """Publish a message to direct and broadcast subscribers."""
        if self._store is not None:
            self._store.save_message(message)

        async with self._lock:
            direct_subscribers = list(self._subscribers.get(message.receiver, []))
            broadcast_subscribers = list(self._subscribers.get("*", []))

        for queue in direct_subscribers + broadcast_subscribers:
            await queue.put(message)
