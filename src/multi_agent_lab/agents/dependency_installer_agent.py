"""Agent that installs workspace dependencies after requirements fixes."""

from __future__ import annotations

import logging

from multi_agent_lab.agents.base_agent import BaseAgent
from multi_agent_lab.core.message import EventType, Message
from multi_agent_lab.tools.command_tool import CommandTool, CommandToolError

logger = logging.getLogger(__name__)


class DependencyInstallerAgent(BaseAgent):
    """Runs controlled dependency installation from requirements.txt."""

    subscribed_events = (EventType.DEPENDENCY_INSTALL_REQUESTED,)

    def __init__(self, name: str, bus, command_tool: CommandTool, event_logger=None) -> None:
        super().__init__(name, bus, event_logger)
        self.command_tool = command_tool

    async def handle_message(self, message: Message) -> None:
        """Install requirements and request a retest when successful."""
        task_id = str(message.content.get("task_id", ""))
        logger.info("Instalando dependencias desde requirements.txt")
        await self.publish(
            EventType.DEPENDENCY_INSTALL_STARTED,
            {"task_id": task_id, "requirements_path": "requirements.txt"},
            source=message,
        )
        try:
            result = self.command_tool.run_pip_install_requirements()
        except CommandToolError as error:
            failure_payload = {
                "task_id": task_id,
                "requirements_path": "requirements.txt",
                "error": str(error),
            }
            await self.publish(
                EventType.DEPENDENCY_INSTALL_FAILED,
                failure_payload,
                source=message,
            )
            await self.publish(
                EventType.WORKFLOW_HALTED,
                {"reason": "dependency_install_failed", **failure_payload},
                source=message,
            )
            return

        payload = {
            "task_id": task_id,
            "requirements_path": "requirements.txt",
            "result": result.to_dict(),
        }
        if not result.success:
            await self.publish(EventType.DEPENDENCY_INSTALL_FAILED, payload, source=message)
            await self.publish(
                EventType.WORKFLOW_HALTED,
                {"reason": "dependency_install_failed", **payload},
                source=message,
            )
            return

        await self.publish(EventType.DEPENDENCY_INSTALL_SUCCEEDED, payload, source=message)
        await self.publish(
            EventType.RETEST_REQUESTED,
            {
                "task_id": str(message.content.get("execution_task_id", task_id)),
                "command_id": message.content.get("command_id", "pytest"),
                "args": list(message.content.get("args", [])),
            },
            source=message,
        )
