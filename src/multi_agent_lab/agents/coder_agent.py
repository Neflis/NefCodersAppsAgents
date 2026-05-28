"""Coder agent that proposes file content for coding tasks."""

from __future__ import annotations

import logging

from multi_agent_lab.agents.base_agent import BaseAgent
from multi_agent_lab.core.agent_event_logger import AgentEventLogger
from multi_agent_lab.core.message import EventType, Message
from multi_agent_lab.core.message_bus import MessageBus
from multi_agent_lab.llm.ollama_client import OllamaClient, OllamaClientError

logger = logging.getLogger(__name__)


class CoderAgent(BaseAgent):
    """Agent that listens for coding tasks and proposes content."""

    subscribed_events = (EventType.TASK_CREATED,)

    def __init__(
        self,
        name: str,
        bus: MessageBus,
        event_logger: AgentEventLogger | None = None,
        ollama_client: OllamaClient | None = None,
        use_ollama: bool = False,
    ) -> None:
        super().__init__(name, bus, event_logger)
        self.ollama_client = ollama_client
        self.use_ollama = use_ollama

    async def handle_message(self, message: Message) -> None:
        """Generate a code proposal for coding tasks."""
        if message.content.get("task_type") != "coding":
            logger.info("Tarea ignorada por tipo: %s", message.content.get("task_type"))
            return

        target_path = str(message.content.get("path", "README.md"))
        title = str(message.content.get("title", "Generar README.md"))
        logger.info(
            "Generando propuesta task_id=%s path=%s", message.content["task_id"], target_path
        )

        proposed_content = self._mock_file_content(title)
        if self.use_ollama and self.ollama_client is not None:
            try:
                proposed_content = await self.ollama_client.generate(title)
            except OllamaClientError as error:
                logger.info("Ollama no disponible: %s", error)

        await self.publish(
            EventType.CODE_PROPOSED,
            {
                "task_id": message.content["task_id"],
                "action": message.content.get("action", "write_file"),
                "path": target_path,
                "content": proposed_content,
            },
            source=message,
        )

    def _mock_file_content(self, title: str) -> str:
        """Return deterministic README content for the demo."""
        return "\n".join(
            [
                "# App de ejemplo",
                "",
                "Proyecto generado por la red multiagente local.",
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
