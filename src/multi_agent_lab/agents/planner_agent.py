"""Planner agent that creates example tasks."""

from __future__ import annotations

import logging

from multi_agent_lab.agents.base_agent import BaseAgent
from multi_agent_lab.core.agent_event_logger import AgentEventLogger
from multi_agent_lab.core.message import Message, MessageType
from multi_agent_lab.core.message_bus import MessageBus
from multi_agent_lab.core.task import Task
from multi_agent_lab.core.task_queue import TaskQueue

logger = logging.getLogger(__name__)


class PlannerAgent(BaseAgent):
    """Agent that creates and tracks planning tasks."""

    def __init__(
        self,
        name: str,
        bus: MessageBus,
        task_queue: TaskQueue,
        event_logger: AgentEventLogger | None = None,
    ) -> None:
        super().__init__(name, bus, event_logger)
        self.task_queue = task_queue

    async def create_example_task(self) -> Task:
        """Create a sample task and notify the coder agent."""
        task = Task(
            title="Generar README.md para una app de ejemplo",
            description="Crear un README breve para una aplicacion de ejemplo.",
            payload={"path": "README.md", "format": "markdown"},
            priority=1,
        )
        await self.task_queue.put(task)
        logger.info("Tarea creada: %s path=%s", task.title, task.payload["path"])
        await self._log_event("task_created", {"task_id": task.id, "title": task.title})
        await self.publish("coder", MessageType.TASK_CREATED, {"task_id": task.id})
        return task

    async def handle_message(self, message: Message) -> None:
        """Handle planner messages."""
        logger.info(
            "Mensaje recibido de %s: %s content=%s",
            message.sender,
            message.type,
            message.content,
        )
