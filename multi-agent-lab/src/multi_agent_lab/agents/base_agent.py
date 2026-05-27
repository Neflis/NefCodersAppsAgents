"""Base async agent implementation."""

from __future__ import annotations

import asyncio
from contextlib import suppress

from multi_agent_lab.core.message import Message
from multi_agent_lab.core.message_bus import MessageBus


class BaseAgent:
    def __init__(self, name: str, bus: MessageBus) -> None:
        self.name = name
        self.bus = bus
        self._running = False
        self._inbox: asyncio.Queue[Message] | None = None
        self._loop_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._inbox = await self.bus.subscribe(self.name)
        self._loop_task = asyncio.create_task(self._run(), name=f"{self.name}-agent")

    async def stop(self) -> None:
        self._running = False
        if self._loop_task is None:
            return
        self._loop_task.cancel()
        with suppress(asyncio.CancelledError):
            await self._loop_task

    async def handle_message(self, message: Message) -> None:
        raise NotImplementedError

    async def publish(self, receiver: str, message_type: str, content: object, priority: int = 0) -> None:
        await self.bus.publish(
            Message(
                sender=self.name,
                receiver=receiver,
                type=message_type,
                content=content,
                priority=priority,
            )
        )

    async def _run(self) -> None:
        if self._inbox is None:
            raise RuntimeError("Agent must be started before running.")

        while self._running:
            message = await self._inbox.get()
            await self.handle_message(message)
