"""Tester agent that simulates validation after file writes."""

from __future__ import annotations

import logging

from multi_agent_lab.agents.base_agent import BaseAgent
from multi_agent_lab.core.message import EventType, Message

logger = logging.getLogger(__name__)


class TesterAgent(BaseAgent):
    """Agent that listens for written files and simulates tests."""

    subscribed_events = (EventType.FILE_WRITTEN,)

    async def handle_message(self, message: Message) -> None:
        """Simulate validation without running commands."""
        path = str(message.content["path"])
        logger.info("Validacion simulada para path=%s", path)
        await self.publish(
            EventType.TEST_PASSED,
            {
                "task_id": message.content["task_id"],
                "path": path,
                "simulated": True,
            },
            source=message,
        )
