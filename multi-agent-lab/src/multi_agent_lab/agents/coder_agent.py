"""Coder agent that simulates handling tasks."""

from __future__ import annotations

from multi_agent_lab.agents.base_agent import BaseAgent
from multi_agent_lab.core.message import Message
from multi_agent_lab.core.message_bus import MessageBus
from multi_agent_lab.core.task import TaskStatus
from multi_agent_lab.core.task_queue import TaskQueue


class CoderAgent(BaseAgent):
    def __init__(self, name: str, bus: MessageBus, task_queue: TaskQueue) -> None:
        super().__init__(name, bus)
        self.task_queue = task_queue

    async def handle_message(self, message: Message) -> None:
        if message.type != "task.created":
            print(f"[coder] mensaje ignorado: {message.type}")
            return

        task = await self.task_queue.get()
        task.status = TaskStatus.IN_PROGRESS
        print(f"[coder] procesando tarea: {task.title}")

        simulated_response = {
            "task_id": task.id,
            "title": task.title,
            "result": "def greet(name: str) -> str: return f'Hola, {name}'",
        }
        task.status = TaskStatus.DONE
        self.task_queue.task_done()

        await self.publish("reviewer", "code.simulated", simulated_response)
