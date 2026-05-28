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
from multi_agent_lab.llm.ollama_client import InvalidJSONError, OllamaClient, OllamaClientError
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
        artifact = str(message.content.get("payload", {}).get("artifact", "readme"))
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
            else self._mock_file_content(title, target_path, artifact)
        )

        result = {"path": target_path, "content": proposed_content}
        if artifact != "design":
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
                current_task={
                    "title": title,
                    "path": target_path,
                    "payload": message.content.get("payload", {}),
                },
                workspace_paths=self._related_paths(target_path),
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
        except (InvalidJSONError, OllamaClientError) as error:
            logger.info("Coder LLM no disponible; usando contenido determinista: %s", error)
            return None

    def _related_paths(self, target_path: str) -> list[str]:
        """Return related files that help preserve project coherence."""
        if target_path == "README.md":
            return ["app.py", "requirements.txt"]
        if target_path == "app.py":
            return ["requirements.txt"]
        if target_path == "requirements.txt":
            return ["app.py"]
        return []

    def _mock_file_content(self, title: str, target_path: str, artifact: str) -> str:
        """Return deterministic content by artifact type."""
        if target_path == "requirements.txt":
            return "Flask>=3.0\n"
        if target_path == "app.py":
            return "\n".join(
                [
                    "from flask import Flask, jsonify, request",
                    "",
                    "app = Flask(__name__)",
                    "todos = []",
                    "",
                    "",
                    "@app.get('/todos')",
                    "def list_todos():",
                    "    return jsonify(todos)",
                    "",
                    "",
                    "@app.post('/todos')",
                    "def create_todo():",
                    "    data = request.get_json(silent=True) or {}",
                    "    todo = {'id': len(todos) + 1, 'title': data.get('title', '')}",
                    "    todos.append(todo)",
                    "    return jsonify(todo), 201",
                    "",
                    "",
                    "if __name__ == '__main__':",
                    "    app.run(debug=True)",
                    "",
                ]
            )
        if artifact == "design":
            return "Estructura: app.py, requirements.txt y README.md para API Flask TODO."
        return "\n".join(
            [
                "# Flask TODO API",
                "",
                "Pequena API Flask para gestionar tareas TODO.",
                "",
                "## Archivos",
                "",
                "- `app.py`: API con endpoints `GET /todos` y `POST /todos`.",
                "- `requirements.txt`: dependencia Flask.",
                "",
                "## Uso",
                "",
                "Instala dependencias y ejecuta `app.py` en un entorno local controlado.",
                "",
                f"Generado para: {title}",
                "",
            ]
        )
