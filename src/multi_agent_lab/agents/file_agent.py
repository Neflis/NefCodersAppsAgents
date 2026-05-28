"""File agent that applies approved file operations."""

from __future__ import annotations

import logging

from multi_agent_lab.agents.base_agent import BaseAgent
from multi_agent_lab.core.agent_event_logger import AgentEventLogger
from multi_agent_lab.core.message import EventType, Message
from multi_agent_lab.core.message_bus import MessageBus
from multi_agent_lab.tools.file_tool import FileTool, FileToolError

logger = logging.getLogger(__name__)


class FileAgent(BaseAgent):
    """Agent that listens for approved write actions and uses FileTool."""

    subscribed_events = (EventType.REVIEW_APPROVED,)

    def __init__(
        self,
        name: str,
        bus: MessageBus,
        file_tool: FileTool,
        event_logger: AgentEventLogger | None = None,
    ) -> None:
        super().__init__(name, bus, event_logger)
        self.file_tool = file_tool

    async def handle_message(self, message: Message) -> None:
        """Write approved content to the workspace."""
        if message.content.get("action") != "write_file":
            return

        path = str(message.content["path"])
        logger.info("Escribiendo archivo aprobado path=%s", path)
        try:
            written_path = self.file_tool.write_file(path, str(message.content["content"]))
        except FileToolError as error:
            await self.publish(
                EventType.FILE_WRITE_FAILED,
                {"task_id": message.content["task_id"], "path": path, "error": str(error)},
                source=message,
            )
            return

        await self.publish(
            EventType.FILE_WRITTEN,
            {"task_id": message.content["task_id"], "path": written_path},
            source=message,
        )
