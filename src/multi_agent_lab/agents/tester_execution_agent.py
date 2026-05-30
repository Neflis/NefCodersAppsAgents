"""Agent that executes whitelisted validation commands."""

from __future__ import annotations

import logging

from multi_agent_lab.agents.base_agent import BaseAgent
from multi_agent_lab.core.capability import Capability
from multi_agent_lab.core.message import EventType, Message
from multi_agent_lab.tools.command_tool import CommandTool, CommandToolError

logger = logging.getLogger(__name__)


class TesterExecutionAgent(BaseAgent):
    """Agent that runs controlled command validations when explicitly enabled."""

    subscribed_events = (
        EventType.TASK_READY,
        EventType.TEST_EXECUTION_REQUESTED,
        EventType.RETEST_REQUESTED,
        EventType.FIX_APPLIED,
    )
    capabilities = (Capability.TESTING_EXECUTION.value,)

    def __init__(self, name: str, bus, command_tool: CommandTool, event_logger=None) -> None:
        super().__init__(name, bus, event_logger)
        self.command_tool = command_tool

    async def handle_message(self, message: Message) -> None:
        """Claim execution tasks or handle execution requests."""
        if message.type == EventType.TASK_READY:
            await self._request_execution(message)
            return
        if message.type == EventType.TEST_EXECUTION_REQUESTED:
            await self._execute(message)
            return
        if message.type == EventType.RETEST_REQUESTED:
            await self._execute(message)
            return
        if message.type == EventType.FIX_APPLIED:
            await self._request_retest(message)

    async def _request_execution(self, message: Message) -> None:
        if not self.can_claim(message):
            return
        await self.claim_task(message)
        payload = dict(message.content.get("payload", {}))
        await self.publish(
            EventType.TEST_EXECUTION_REQUESTED,
            {
                "task_id": message.content["task_id"],
                "command_id": payload.get("command_id", "pytest"),
                "args": list(payload.get("args", [])),
            },
            source=message,
        )

    async def _execute(self, message: Message) -> None:
        task_id = str(message.content["task_id"])
        command_id = str(message.content.get("command_id", "pytest"))
        args = list(message.content.get("args", []))
        logger.info("Ejecutando validacion command=%s args=%s", command_id, args)
        await self.publish(
            EventType.TEST_EXECUTION_STARTED,
            {"task_id": task_id, "command_id": command_id, "args": args},
            source=message,
        )
        try:
            result = self._run_controlled_command(command_id, args)
        except CommandToolError as error:
            await self._publish_failed(
                message,
                task_id,
                {
                    "task_id": task_id,
                    "command": " ".join([command_id, *[str(arg) for arg in args]]).strip(),
                    "command_id": command_id,
                    "args": args,
                    "exit_code": -1,
                    "stdout": "",
                    "stderr": str(error),
                    "failed_files": [],
                    "suggested_focus_files": ["app.py", "tests/test_app.py"],
                    "error": str(error),
                },
            )
            return

        payload = self._execution_payload(task_id, command_id, args, result.to_dict())
        if result.success:
            await self.publish(EventType.TEST_EXECUTION_PASSED, payload, source=message)
            await self.publish(
                EventType.TASK_COMPLETED,
                {"task_id": task_id, "result": payload, "owner": self.name},
                source=message,
            )
            return

        await self._publish_failed(message, task_id, payload)

    def _run_controlled_command(self, command_id: str, args: list[object]):
        """Run validation commands through their safest CommandTool entrypoints."""
        normalized_args = [str(arg) for arg in args]
        if command_id == "pytest":
            return self.command_tool.run_pytest(normalized_args)
        return self.command_tool.run_command(command_id, normalized_args)

    async def _publish_failed(
        self,
        message: Message,
        task_id: str,
        payload: dict[str, object],
    ) -> None:
        await self.publish(EventType.TEST_EXECUTION_FAILED, payload, source=message)
        error = self._feedback_from_failure(payload)
        if message.type in {EventType.TEST_EXECUTION_REQUESTED, EventType.RETEST_REQUESTED}:
            return
        await self.publish(
            EventType.TASK_FAILED,
            {"task_id": task_id, "error": error, "execution_feedback": payload},
            source=message,
        )

    async def _request_retest(self, message: Message) -> None:
        if (
            message.type == EventType.FIX_APPLIED
            and message.content.get("path") == "requirements.txt"
        ):
            return
        command_id = str(message.content.get("command_id", "pytest"))
        args = list(message.content.get("args", []))
        task_id = str(message.content.get("execution_task_id", message.content.get("task_id")))
        await self.publish(
            EventType.RETEST_REQUESTED,
            {"task_id": task_id, "command_id": command_id, "args": args},
            source=message,
        )

    def _execution_payload(
        self,
        task_id: str,
        command_id: str,
        args: list[object],
        result: dict[str, object],
    ) -> dict[str, object]:
        """Build rich execution event content."""
        stdout = str(result.get("stdout", ""))
        stderr = str(result.get("stderr", ""))
        return {
            "task_id": task_id,
            "command": " ".join([command_id, *[str(arg) for arg in args]]).strip(),
            "command_id": command_id,
            "args": args,
            "exit_code": result.get("exit_code", -1),
            "stdout": stdout,
            "stderr": stderr,
            "failed_files": self._infer_failed_files(stdout, stderr),
            "suggested_focus_files": self._suggest_focus_files(stdout, stderr),
            "result": result,
        }

    def _infer_failed_files(self, stdout: str, stderr: str) -> list[str]:
        """Infer likely failed files from command output."""
        combined = f"{stdout}\n{stderr}"
        candidates = []
        for path in ("app.py", "requirements.txt", "README.md", "tests/test_app.py"):
            if path in combined:
                candidates.append(path)
        return candidates

    def _suggest_focus_files(self, stdout: str, stderr: str) -> list[str]:
        """Suggest safe files for a coder fix task."""
        combined = f"{stdout}\n{stderr}"
        if "No module named 'app'" in combined or 'No module named "app"' in combined:
            return ["tests/test_app.py", "app.py"]
        if "requirements" in combined or "ModuleNotFoundError" in combined:
            return ["requirements.txt", "app.py"]
        if "README" in combined:
            return ["README.md", "tests/test_app.py"]
        if "SyntaxError" in combined:
            return ["app.py", "tests/test_app.py"]
        return self._infer_failed_files(stdout, stderr) or ["app.py", "tests/test_app.py"]

    def _feedback_from_failure(self, payload: dict[str, object]) -> str:
        """Create actionable feedback from command failure output."""
        result = payload.get("result")
        if not isinstance(result, dict):
            return str(payload.get("error", "Command execution failed."))
        stderr = str(result.get("stderr", ""))
        stdout = str(result.get("stdout", ""))
        combined = f"{stderr}\n{stdout}"
        if "ModuleNotFoundError" in combined or "ImportError" in combined:
            return f"Missing import or dependency during test execution: {combined[:1000]}"
        if "SyntaxError" in combined:
            return f"Syntax error during test execution: {combined[:1000]}"
        if result.get("timed_out"):
            return "Command execution timed out."
        return f"Command execution failed: {combined[:1000]}"
