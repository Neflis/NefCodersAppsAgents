"""Tester agent that simulates validation after file writes."""

from __future__ import annotations

import logging

from multi_agent_lab.agents.base_agent import BaseAgent
from multi_agent_lab.core.capability import Capability
from multi_agent_lab.core.message import EventType, Message
from multi_agent_lab.tools.file_tool import FileTool

logger = logging.getLogger(__name__)


class TesterAgent(BaseAgent):
    """Agent that claims mock testing tasks and simulates validation."""

    subscribed_events = (EventType.TASK_READY,)
    capabilities = (Capability.TESTING_MOCK.value,)

    def __init__(self, name: str, bus, file_tool: FileTool, event_logger=None) -> None:
        super().__init__(name, bus, event_logger)
        self.file_tool = file_tool

    async def handle_message(self, message: Message) -> None:
        """Claim compatible testing tasks and simulate validation."""
        if not self.can_claim(message):
            return
        await self.claim_task(message)

        payload = message.content.get("payload", {})
        paths = list(payload.get("paths", [])) or [str(payload.get("path", "README.md"))]
        logger.info("Validacion simulada task_id=%s paths=%s", message.content["task_id"], paths)
        missing = [path for path in paths if not self.file_tool.exists(path)]
        if missing:
            await self.publish(
                EventType.TEST_FAILED,
                {
                    "task_id": message.content["task_id"],
                    "paths": missing,
                    "error": "Archivo no existe",
                },
                source=message,
            )
            await self.publish(
                EventType.TASK_FAILED,
                {"task_id": message.content["task_id"], "error": "Archivo no existe"},
                source=message,
            )
            return

        result = {"paths": paths, "simulated": True}
        await self.publish(
            EventType.TEST_PASSED,
            {"task_id": message.content["task_id"], **result},
            source=message,
        )
        await self.publish(
            EventType.TASK_COMPLETED,
            {"task_id": message.content["task_id"], "result": result, "owner": self.name},
            source=message,
        )
