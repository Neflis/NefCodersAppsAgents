"""Coder agent that proposes content for coding tasks."""

from __future__ import annotations

import logging

from multi_agent_lab.agents.base_agent import BaseAgent
from multi_agent_lab.core.agent_event_logger import AgentEventLogger
from multi_agent_lab.core.capability import Capability
from multi_agent_lab.core.message import EventType, Message
from multi_agent_lab.core.message_bus import MessageBus
from multi_agent_lab.llm.ollama_client import OllamaClient, OllamaClientError

logger = logging.getLogger(__name__)


class CoderAgent(BaseAgent):
    """Agent that claims coding tasks and proposes content."""

    subscribed_events = (EventType.TASK_READY,)
    capabilities = (Capability.CODING.value,)

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
        """Claim compatible coding tasks and complete them with a proposal."""
        if not self.can_claim(message):
            return
        await self.claim_task(message)

        target_path = str(message.content.get("payload", {}).get("path", "README.md"))
        title = str(message.content.get("title", "Crear borrador README"))
        logger.info(
            "Generando borrador task_id=%s path=%s",
            message.content["task_id"],
            target_path,
        )

        proposed_content = self._mock_file_content(title)
        if self.use_ollama and self.ollama_client is not None:
            try:
                proposed_content = await self.ollama_client.generate(title)
            except OllamaClientError as error:
                logger.info("Ollama no disponible: %s", error)

        result = {"path": target_path, "content": proposed_content}
        await self.publish(
            EventType.CODE_PROPOSED,
            {"task_id": message.content["task_id"], **result},
            source=message,
        )
        await self.publish(
            EventType.TASK_COMPLETED,
            {"task_id": message.content["task_id"], "result": result, "owner": self.name},
            source=message,
        )

    def _mock_file_content(self, title: str) -> str:
        """Return deterministic README content for the TODO demo."""
        return "\n".join(
            [
                "# TODO App",
                "",
                "Documentacion inicial para una pequena aplicacion TODO.",
                "",
                "## Objetivo",
                "",
                title,
                "",
                "## Funcionalidades previstas",
                "",
                "- Crear tareas",
                "- Marcar tareas como completadas",
                "- Listar tareas pendientes",
                "",
            ]
        )
