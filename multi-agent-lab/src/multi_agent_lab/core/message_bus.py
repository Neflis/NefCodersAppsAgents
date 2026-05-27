"""Async in-memory message bus."""

from __future__ import annotations

import asyncio
from collections import defaultdict

from multi_agent_lab.core.message import Message


class MessageBus:
    def __init__(self) -> None:
        self._subscribers: dict[str, list[asyncio.Queue[Message]]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def subscribe(self, receiver: str) -> asyncio.Queue[Message]:
        queue: asyncio.Queue[Message] = asyncio.Queue()
        async with self._lock:
            self._subscribers[receiver].append(queue)
        return queue

    async def publish(self, message: Message) -> None:
        async with self._lock:
            direct_subscribers = list(self._subscribers.get(message.receiver, []))
            broadcast_subscribers = list(self._subscribers.get("*", []))

        for queue in direct_subscribers + broadcast_subscribers:
            await queue.put(message)
