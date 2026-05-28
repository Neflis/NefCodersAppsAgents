"""Reviewer agent that validates proposed content."""

from __future__ import annotations

import logging

from multi_agent_lab.agents.base_agent import BaseAgent
from multi_agent_lab.core.capability import Capability
from multi_agent_lab.core.file_awareness import FileAwarenessService
from multi_agent_lab.core.message import EventType, Message
from multi_agent_lab.core.task_graph_store import TaskGraphStore
from multi_agent_lab.llm.context_builder import AgentContextBuilder
from multi_agent_lab.llm.decision import LLMDecision
from multi_agent_lab.llm.ollama_client import InvalidJSONError, OllamaClient, OllamaClientError
from multi_agent_lab.llm.prompt_template import PromptTemplate

logger = logging.getLogger(__name__)


class ReviewerAgent(BaseAgent):
    """Agent that claims review tasks and approves or rejects content."""

    subscribed_events = (EventType.TASK_READY,)
    capabilities = (Capability.REVIEWING.value,)

    def __init__(
        self,
        name: str,
        bus,
        graph_store: TaskGraphStore,
        event_logger=None,
        llm_client: OllamaClient | None = None,
        context_builder: AgentContextBuilder | None = None,
        file_awareness: FileAwarenessService | None = None,
    ) -> None:
        super().__init__(name, bus, event_logger)
        self.graph_store = graph_store
        self.llm_client = llm_client
        self.context_builder = context_builder or AgentContextBuilder(graph_store)
        self.file_awareness = file_awareness

    async def handle_message(self, message: Message) -> None:
        """Claim compatible review tasks and complete them."""
        if not self.can_claim(message):
            return
        await self.claim_task(message)

        if message.content.get("payload", {}).get("project_review"):
            await self._review_project(message)
            return

        graph = self.graph_store.get(message.correlation_id or "")
        dependency_results = graph.dependency_results(str(message.content["task_id"]))
        draft = dependency_results[-1] if dependency_results else {}
        content = str(draft.get("content", ""))
        path = str(
            draft.get(
                "path",
                message.content.get("payload", {}).get("path", "README.md"),
            )
        )
        logger.info("Revisando contenido task_id=%s path=%s", message.content["task_id"], path)

        decision = await self._decide(message, content, path)
        approved = self._is_valid_content(content)
        if decision is not None:
            approved = decision.action == "approve"

        if not approved:
            await self.publish(
                EventType.REVIEW_REJECTED,
                {
                    "task_id": message.content["task_id"],
                    "path": path,
                    "reason": "Contenido invalido",
                },
                source=message,
            )
            await self.publish(
                EventType.TASK_FAILED,
                {"task_id": message.content["task_id"], "error": "Contenido invalido"},
                source=message,
            )
            return

        result = {"path": path, "content": content, "approved": True}
        await self.publish(
            EventType.REVIEW_APPROVED,
            {"task_id": message.content["task_id"], **result},
            source=message,
        )
        await self.publish(
            EventType.TASK_COMPLETED,
            {"task_id": message.content["task_id"], "result": result, "owner": self.name},
            source=message,
        )

    async def _review_project(self, message: Message) -> None:
        """Review final multi-file project coherence."""
        paths = list(message.content.get("payload", {}).get("paths", []))
        files = self.file_awareness.read_relevant_files(paths) if self.file_awareness else {}
        feedback = self._project_feedback(files)
        if feedback:
            await self.publish(
                EventType.PROJECT_REVIEW_REJECTED,
                {"task_id": message.content["task_id"], "feedback": feedback},
                source=message,
            )
            await self.publish(
                EventType.TASK_FAILED,
                {"task_id": message.content["task_id"], "error": "; ".join(feedback)},
                source=message,
            )
            return

        result = {"paths": paths, "approved": True}
        await self.publish(
            EventType.PROJECT_REVIEW_APPROVED,
            {"task_id": message.content["task_id"], **result},
            source=message,
        )
        await self.publish(
            EventType.TASK_COMPLETED,
            {"task_id": message.content["task_id"], "result": result, "owner": self.name},
            source=message,
        )

    def _project_feedback(self, files: dict[str, str]) -> list[str]:
        """Return actionable feedback for a multi-file Flask project."""
        feedback: list[str] = []
        app = files.get("app.py", "")
        requirements = files.get("requirements.txt", "")
        readme = files.get("README.md", "")
        if "from flask import" in app.lower() and "flask" not in requirements.lower():
            feedback.append("requirements.txt debe incluir Flask.")
        if "flask" not in readme.lower() or "todo" not in readme.lower():
            feedback.append("README.md debe describir la API Flask TODO.")
        if "GET /todos" not in readme and "/todos" not in readme:
            feedback.append("README.md debe mencionar los endpoints TODO.")
        return feedback

    def _is_valid_content(self, content: str) -> bool:
        """Validate generated content before approving file writing."""
        return bool(content.strip()) and len(content.encode("utf-8")) <= 1024 * 1024

    async def _decide(self, message: Message, content: str, path: str) -> LLMDecision | None:
        """Ask the LLM to approve or reject content."""
        if self.llm_client is None:
            return None
        self.context_builder.record_event(message)
        prompt = PromptTemplate(
            identity="ReviewerAgent",
            capabilities=[Capability.REVIEWING.value],
            constraints=["Return JSON only.", "Approve only safe non-empty content."],
            input_context=self.context_builder.build(
                message,
                current_task={"path": path, "content": content},
            ),
            expected_json_output={
                "action": "approve",
                "reasoning_summary": "short text",
                "confidence": 0.0,
                "content": {"approved": True},
                "events_to_publish": [],
                "task_updates": [],
            },
        ).render()
        try:
            return LLMDecision.from_dict(await self.llm_client.generate_json(prompt))
        except (InvalidJSONError, OllamaClientError) as error:
            logger.info("Reviewer LLM no disponible; usando validacion determinista: %s", error)
            return None
