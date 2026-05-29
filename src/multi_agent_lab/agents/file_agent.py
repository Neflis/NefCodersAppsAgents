"""File agent that applies approved file operations."""

from __future__ import annotations

import logging

from multi_agent_lab.agents.base_agent import BaseAgent
from multi_agent_lab.core.agent_event_logger import AgentEventLogger
from multi_agent_lab.core.capability import Capability
from multi_agent_lab.core.message import EventType, Message
from multi_agent_lab.core.message_bus import MessageBus
from multi_agent_lab.core.task_graph_store import TaskGraphStore
from multi_agent_lab.tools.file_tool import FileTool, FileToolError

logger = logging.getLogger(__name__)


class FileAgent(BaseAgent):
    """Agent that claims file write tasks and uses FileTool."""

    subscribed_events = (EventType.TASK_READY, EventType.CODE_PROPOSED, EventType.FIX_PROPOSED)
    capabilities = (Capability.FILE_WRITE.value,)

    def __init__(
        self,
        name: str,
        bus: MessageBus,
        file_tool: FileTool,
        graph_store: TaskGraphStore,
        event_logger: AgentEventLogger | None = None,
    ) -> None:
        super().__init__(name, bus, event_logger)
        self.file_tool = file_tool
        self.graph_store = graph_store

    async def handle_message(self, message: Message) -> None:
        """Claim compatible file tasks and write approved content."""
        if message.type == EventType.CODE_PROPOSED:
            await self._write_proposed_file(message)
            return
        if message.type == EventType.FIX_PROPOSED:
            await self._apply_fix(message)
            return
        if not self.can_claim(message):
            return
        await self.claim_task(message)

        graph = self.graph_store.get(message.correlation_id or "")
        dependency_results = graph.dependency_results(str(message.content["task_id"]))
        approved = dependency_results[-1] if dependency_results else {}
        path = str(
            approved.get(
                "path",
                message.content.get("payload", {}).get("path", "README.md"),
            )
        )
        content = str(approved.get("content", ""))
        logger.info("Escribiendo archivo task_id=%s path=%s", message.content["task_id"], path)
        try:
            written_path = self.file_tool.write_file(path, content)
        except FileToolError as error:
            await self.publish(
                EventType.FILE_WRITE_FAILED,
                {"task_id": message.content["task_id"], "path": path, "error": str(error)},
                source=message,
            )
            await self.publish(
                EventType.TASK_FAILED,
                {"task_id": message.content["task_id"], "error": str(error)},
                source=message,
            )
            return

        result = {"path": written_path}
        await self.publish(
            EventType.FILE_WRITTEN,
            {"task_id": message.content["task_id"], **result},
            source=message,
        )
        await self.publish(
            EventType.TASK_COMPLETED,
            {"task_id": message.content["task_id"], "result": result, "owner": self.name},
            source=message,
        )

    async def _write_proposed_file(self, message: Message) -> None:
        """Write a proposed file emitted by a coding task."""
        path = str(message.content["path"])
        logger.info("Escribiendo propuesta path=%s", path)
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

    async def _apply_fix(self, message: Message) -> None:
        """Apply a proposed fix as a safe file replacement."""
        path = str(message.content["path"])
        logger.info("Aplicando fix path=%s", path)
        try:
            written_path = self.file_tool.write_file(path, str(message.content["content"]))
        except FileToolError as error:
            await self.publish(
                EventType.FIX_FAILED,
                {
                    "task_id": message.content["task_id"],
                    "path": path,
                    "error": str(error),
                    "execution_task_id": message.content.get("execution_task_id"),
                },
                source=message,
            )
            await self.publish(
                EventType.TASK_FAILED,
                {"task_id": message.content["task_id"], "error": str(error)},
                source=message,
            )
            return
        await self.publish(
            EventType.FIX_APPLIED,
            {
                "task_id": message.content["task_id"],
                "path": written_path,
                "reason": message.content.get("reason", ""),
                "based_on_error": message.content.get("based_on_error", ""),
                "fix_reasoning": message.content.get("fix_reasoning", ""),
                "diff_summary": message.content.get("diff_summary", ""),
                "failure_context": message.content.get("failure_context", {}),
                "execution_task_id": message.content.get("execution_task_id"),
                "command_id": message.content.get("command_id", "pytest"),
                "args": list(message.content.get("args", [])),
            },
            source=message,
        )
