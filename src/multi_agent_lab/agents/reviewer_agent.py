"""Reviewer agent that reviews simulated coder output."""

from __future__ import annotations

import logging

from multi_agent_lab.agents.base_agent import BaseAgent
from multi_agent_lab.core.agent_event_logger import AgentEventLogger
from multi_agent_lab.core.message import Message, MessageType
from multi_agent_lab.core.message_bus import MessageBus
from multi_agent_lab.tools.file_tool import FileTool, FileToolError

logger = logging.getLogger(__name__)


class ReviewerAgent(BaseAgent):
    """Agent that reviews coder responses."""

    def __init__(
        self,
        name: str,
        bus: MessageBus,
        event_logger: AgentEventLogger | None = None,
        file_tool: FileTool | None = None,
    ) -> None:
        super().__init__(name, bus, event_logger)
        self.file_tool = file_tool

    async def handle_message(self, message: Message) -> None:
        """Review simulated code messages."""
        if message.type != MessageType.FILE_WRITE_REQUEST:
            logger.info("Mensaje ignorado: %s", message.type)
            return

        content = message.content
        target_path = content["path"]
        logger.info("Revisando escritura task_id=%s path=%s", content["task_id"], target_path)

        if self.file_tool is None:
            await self.publish(
                "planner",
                MessageType.TASK_FAILED,
                {"task_id": content["task_id"], "error": "FileTool no configurado."},
            )
            return

        if not self._is_valid_content(content["content"]):
            await self.publish(
                "planner",
                MessageType.TASK_FAILED,
                {"task_id": content["task_id"], "error": "Contenido rechazado por reviewer."},
            )
            return

        try:
            written_path = self.file_tool.write_file(target_path, content["content"])
        except FileToolError as error:
            await self.publish(
                "planner",
                MessageType.TASK_FAILED,
                {"task_id": content["task_id"], "path": target_path, "error": str(error)},
            )
            return

        logger.info("Archivo modificado path=%s", written_path)
        await self.publish(
            "planner",
            MessageType.FILE_WRITE_RESULT,
            {
                "task_id": content["task_id"],
                "path": written_path,
                "approved": True,
            },
        )

    def _is_valid_content(self, content: str) -> bool:
        """Validate generated content before file writing."""
        return bool(content.strip()) and len(content.encode("utf-8")) <= 1024 * 1024
