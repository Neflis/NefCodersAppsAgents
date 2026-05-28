"""Reviewer agent that validates proposed content."""

from __future__ import annotations

import logging

from multi_agent_lab.agents.base_agent import BaseAgent
from multi_agent_lab.core.capability import Capability
from multi_agent_lab.core.message import EventType, Message
from multi_agent_lab.core.task_graph_store import TaskGraphStore

logger = logging.getLogger(__name__)


class ReviewerAgent(BaseAgent):
    """Agent that claims review tasks and approves or rejects content."""

    subscribed_events = (EventType.TASK_READY,)
    capabilities = (Capability.REVIEWING.value,)

    def __init__(self, name: str, bus, graph_store: TaskGraphStore, event_logger=None) -> None:
        super().__init__(name, bus, event_logger)
        self.graph_store = graph_store

    async def handle_message(self, message: Message) -> None:
        """Claim compatible review tasks and complete them."""
        if not self.can_claim(message):
            return
        await self.claim_task(message)

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

        if not self._is_valid_content(content):
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

    def _is_valid_content(self, content: str) -> bool:
        """Validate generated content before approving file writing."""
        return bool(content.strip()) and len(content.encode("utf-8")) <= 1024 * 1024
