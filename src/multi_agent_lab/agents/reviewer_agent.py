"""Reviewer agent that reviews simulated coder output."""

from __future__ import annotations

import logging

from multi_agent_lab.agents.base_agent import BaseAgent
from multi_agent_lab.core.message import Message

logger = logging.getLogger(__name__)


class ReviewerAgent(BaseAgent):
    """Agent that reviews coder responses."""

    async def handle_message(self, message: Message) -> None:
        """Review simulated code messages."""
        if message.type != "code.simulated":
            logger.info("Mensaje ignorado: %s", message.type)
            return

        content = message.content
        logger.info("Revisando tarea %s", content["task_id"])
        review = {
            "task_id": content["task_id"],
            "approved": True,
            "notes": "Respuesta simulada aceptada para la demo asincrona.",
        }
        await self.publish("planner", "review.completed", review)
