"""Coder agent that proposes content for coding tasks."""

from __future__ import annotations

import logging

from multi_agent_lab.agents.base_agent import BaseAgent
from multi_agent_lab.core.agent_event_logger import AgentEventLogger
from multi_agent_lab.core.capability import Capability
from multi_agent_lab.core.message import EventType, Message
from multi_agent_lab.core.message_bus import MessageBus
from multi_agent_lab.llm.context_builder import AgentContextBuilder
from multi_agent_lab.llm.decision import LLMDecision
from multi_agent_lab.llm.ollama_client import InvalidJSONError, OllamaClient
from multi_agent_lab.llm.prompt_template import PromptTemplate

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
        context_builder: AgentContextBuilder | None = None,
    ) -> None:
        super().__init__(name, bus, event_logger)
        self.ollama_client = ollama_client
        self.context_builder = context_builder or AgentContextBuilder()

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

        decision = await self._decide(message, title, target_path)
        proposed_content = (
            str(decision.content)
            if decision is not None and isinstance(decision.content, str)
            else self._mock_file_content(title)
        )

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

    async def _decide(self, message: Message, title: str, target_path: str) -> LLMDecision | None:
        """Ask the LLM to generate task content."""
        if self.ollama_client is None:
            return None
        self.context_builder.record_event(message)
        prompt = PromptTemplate(
            identity="CoderAgent",
            capabilities=[Capability.CODING.value],
            constraints=["Return JSON only.", "Generate content only for workspace files."],
            input_context=self.context_builder.build(
                message,
                current_task={"title": title, "path": target_path},
            ),
            expected_json_output={
                "action": "generate_content",
                "reasoning_summary": "short text",
                "confidence": 0.0,
                "content": "markdown content",
                "events_to_publish": [],
                "task_updates": [],
            },
        ).render()
        try:
            return LLMDecision.from_dict(await self.ollama_client.generate_json(prompt))
        except InvalidJSONError as error:
            logger.info("Coder LLM JSON invalido; usando contenido determinista: %s", error)
            return None

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
