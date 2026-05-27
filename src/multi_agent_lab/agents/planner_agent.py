"""Planner agent that creates example tasks."""

from __future__ import annotations

from multi_agent_lab.agents.base_agent import BaseAgent
from multi_agent_lab.core.agent_event_logger import AgentEventLogger
from multi_agent_lab.core.message import Message
from multi_agent_lab.core.message_bus import MessageBus
from multi_agent_lab.core.task import Task
from multi_agent_lab.core.task_queue import TaskQueue


class PlannerAgent(BaseAgent):
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
        task = Task(
            title="Crear funcion de saludo",
            description="Simular la creacion de una funcion Python que saluda por nombre.",
            priority=1,
        )
        await self.task_queue.put(task)
        print(f"[planner] tarea creada: {task.title}")
        await self._log_event("task_created", {"task_id": task.id, "title": task.title})
        await self.publish("coder", "task.created", {"task_id": task.id})
        return task

    async def handle_message(self, message: Message) -> None:
        print(f"[planner] mensaje recibido de {message.sender}: {message.type}")
