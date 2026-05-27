"""Coder agent that simulates or requests task handling."""

from __future__ import annotations

import logging

from multi_agent_lab.agents.base_agent import BaseAgent
from multi_agent_lab.core.agent_event_logger import AgentEventLogger
from multi_agent_lab.core.message import Message, MessageType
from multi_agent_lab.core.message_bus import MessageBus
from multi_agent_lab.core.task import TaskStatus
from multi_agent_lab.core.task_queue import TaskQueue
from multi_agent_lab.llm.ollama_client import OllamaClient, OllamaClientError

logger = logging.getLogger(__name__)


class CoderAgent(BaseAgent):
    """Agent that simulates coding or delegates generation to Ollama."""

    def __init__(
        self,
        name: str,
        bus: MessageBus,
        task_queue: TaskQueue,
        event_logger: AgentEventLogger | None = None,
        ollama_client: OllamaClient | None = None,
        use_ollama: bool = False,
    ) -> None:
        super().__init__(name, bus, event_logger)
        self.task_queue = task_queue
        self.ollama_client = ollama_client
        self.use_ollama = use_ollama

    async def handle_message(self, message: Message) -> None:
        """Process task messages and publish a code response."""
        if message.type != MessageType.TASK_CREATED:
            logger.info("Mensaje ignorado: %s", message.type)
            return

        task = await self.task_queue.get()
        await self.task_queue.update_status(task, TaskStatus.IN_PROGRESS)
        target_path = str(task.payload.get("path", "README.md"))
        logger.info("Procesando tarea: %s path=%s", task.title, target_path)

        result = self._mock_file_content(task.title)
        if self.use_ollama and self.ollama_client is not None:
            try:
                result = await self.ollama_client.generate(task.description)
            except OllamaClientError as error:
                result = f"Ollama no disponible: {error}"

        response = {
            "task_id": task.id,
            "title": task.title,
            "path": target_path,
            "content": result,
        }
        await self.task_queue.update_status(task, TaskStatus.DONE)
        await self._log_event("task_completed", {"task_id": task.id, "title": task.title})
        self.task_queue.task_done()

        await self.publish("reviewer", MessageType.FILE_WRITE_REQUEST, response)

    def _mock_file_content(self, title: str) -> str:
        """Return deterministic README content for the demo."""
        return "\n".join(
            [
                "# App de ejemplo",
                "",
                "Proyecto generado por el laboratorio multiagente local.",
                "",
                "## Objetivo",
                "",
                title,
                "",
                "## Uso",
                "",
                "Describe aqui como ejecutar la app cuando exista implementacion.",
                "",
            ]
        )
