"""Reviewer agent that approves or rejects proposed content."""

from __future__ import annotations

import logging

from multi_agent_lab.agents.base_agent import BaseAgent
from multi_agent_lab.core.message import EventType, Message

logger = logging.getLogger(__name__)


class ReviewerAgent(BaseAgent):
    """Agent that listens for code proposals and emits review events."""

    subscribed_events = (EventType.CODE_PROPOSED,)

    async def handle_message(self, message: Message) -> None:
        """Review proposed content without writing files directly."""
        content = str(message.content.get("content", ""))
        target_path = str(message.content.get("path", "README.md"))
        logger.info(
            "Revisando propuesta task_id=%s path=%s", message.content["task_id"], target_path
        )

        if not self._is_valid_content(content):
            await self.publish(
                EventType.REVIEW_REJECTED,
                {
                    "task_id": message.content["task_id"],
                    "path": target_path,
                    "reason": "Contenido vacio o demasiado grande.",
                },
                source=message,
            )
            return

        await self.publish(
            EventType.REVIEW_APPROVED,
            {
                "task_id": message.content["task_id"],
                "action": message.content.get("action", "write_file"),
                "path": target_path,
                "content": content,
            },
            source=message,
        )

    def _is_valid_content(self, content: str) -> bool:
        """Validate generated content before approving file writing."""
        return bool(content.strip()) and len(content.encode("utf-8")) <= 1024 * 1024
