"""Base async agent implementation."""

from __future__ import annotations

import asyncio
from contextlib import suppress

from multi_agent_lab.core.agent_event_logger import AgentEventLogger
from multi_agent_lab.core.message import EventType, Message
from multi_agent_lab.core.message_bus import MessageBus


class BaseAgent:
    """Base class for autonomous event-driven agents."""

    subscribed_events: tuple[str | EventType, ...] = ()
    capabilities: tuple[str, ...] = ()

    def __init__(
        self, name: str, bus: MessageBus, event_logger: AgentEventLogger | None = None
    ) -> None:
        self.name = name
        self.bus = bus
        self.event_logger = event_logger
        self.claimed_tasks: set[str] = set()
        self._running = False
        self._inbox: asyncio.Queue[Message] | None = None
        self._loop_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Subscribe the agent to its events and start its message loop."""
        if self._running:
            return
        self._running = True
        self._inbox = await self.bus.subscribe_many(self.subscribed_events)
        self._loop_task = asyncio.create_task(self._run(), name=f"{self.name}-agent")
        await self._log_event("agent_started", {"events": [str(e) for e in self.subscribed_events]})

    def can_claim(self, message: Message) -> bool:
        """Return whether this agent can claim a ready task event."""
        task_id = str(message.content.get("task_id", ""))
        required_capability = str(message.content.get("required_capability", ""))
        return (
            task_id not in self.claimed_tasks
            and required_capability in self.capabilities
            and message.content.get("status") == "READY"
        )

    async def claim_task(self, message: Message) -> None:
        """Mark a task as locally claimed and publish TASK_CLAIMED."""
        task_id = str(message.content["task_id"])
        self.claimed_tasks.add(task_id)
        await self.publish(
            EventType.TASK_CLAIMED,
            {
                "task_id": task_id,
                "required_capability": message.content["required_capability"],
                "owner": self.name,
            },
            source=message,
        )

    async def stop(self) -> None:
        """Stop the agent message loop."""
        self._running = False
        if self._loop_task is None:
            return
        self._loop_task.cancel()
        with suppress(asyncio.CancelledError):
            await self._loop_task
        await self._log_event("agent_stopped")

    async def handle_message(self, message: Message) -> None:
        """Handle one incoming event."""
        raise NotImplementedError

    async def publish(
        self,
        event_type: str | EventType,
        content: object,
        source: Message | None = None,
        priority: int = 0,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Publish an event caused by an optional source event."""
        await self.bus.publish(
            Message(
                sender=self.name,
                type=event_type,
                content=content,
                priority=priority,
                correlation_id=source.correlation_id if source else None,
                causation_id=source.id if source else None,
                metadata=metadata or {},
            )
        )

    async def _log_event(self, event_type: str, details: dict[str, object] | None = None) -> None:
        if self.event_logger is not None:
            await self.event_logger.log(self.name, event_type, details)

    async def _run(self) -> None:
        if self._inbox is None:
            raise RuntimeError("Agent must be started before running.")

        while self._running:
            message = await self._inbox.get()
            await self._log_event(
                "message_received",
                {
                    "message_id": message.id,
                    "sender": message.sender,
                    "type": str(message.type),
                    "correlation_id": message.correlation_id,
                },
            )
            await self.handle_message(message)
            await self._log_event(
                "message_processed",
                {
                    "message_id": message.id,
                    "sender": message.sender,
                    "type": str(message.type),
                    "correlation_id": message.correlation_id,
                },
            )
