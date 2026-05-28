"""Planner agent that turns goals into tasks."""

from __future__ import annotations

import logging

from multi_agent_lab.agents.base_agent import BaseAgent
from multi_agent_lab.core.agent_event_logger import AgentEventLogger
from multi_agent_lab.core.message import EventType, Message
from multi_agent_lab.core.message_bus import MessageBus
from multi_agent_lab.core.task import Task
from multi_agent_lab.core.task_queue import TaskQueue

logger = logging.getLogger(__name__)


class PlannerAgent(BaseAgent):
    """Agent that listens for goals and publishes task events."""

    subscribed_events = (EventType.GOAL_SUBMITTED,)

    def __init__(
        self,
        name: str,
        bus: MessageBus,
        task_queue: TaskQueue,
        event_logger: AgentEventLogger | None = None,
    ) -> None:
        super().__init__(name, bus, event_logger)
        self.task_queue = task_queue

    async def handle_message(self, message: Message) -> None:
        """Create a coding task from a submitted goal."""
        goal = str(message.content.get("goal", "Generar README.md para una app de ejemplo"))
        path = str(message.content.get("path", "README.md"))
        task = Task(
            title=goal,
            description="Crear contenido para el archivo solicitado.",
            payload={"path": path, "task_type": "coding", "action": "write_file"},
            priority=1,
        )
        await self.task_queue.put(task)
        logger.info("Tarea creada task_id=%s path=%s", task.id, path)
        await self._log_event("task_created", {"task_id": task.id, "title": task.title})
        await self.publish(
            EventType.TASK_CREATED,
            {
                "task_id": task.id,
                "task_type": "coding",
                "action": "write_file",
                "title": task.title,
                "description": task.description,
                "path": path,
            },
            source=message,
        )
