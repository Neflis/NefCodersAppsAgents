"""Controlled command execution restricted to the workspace."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from multi_agent_lab.core.workspace_manager import WorkspaceManager, WorkspaceSecurityError


class CommandToolError(ValueError):
    """Raised when a command is not allowed or cannot run safely."""


@dataclass(frozen=True, slots=True)
class CommandExecutionResult:
    """Result of one controlled command execution."""

    success: bool
    exit_code: int
    stdout: str
    stderr: str
    duration: float
    timed_out: bool = False

    def to_dict(self) -> dict[str, object]:
        """Serialize command result."""
        return {
            "success": self.success,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration": self.duration,
            "timed_out": self.timed_out,
        }


class CommandTool:
    """Run a strict whitelist of commands inside the workspace."""

    allowed_commands = {"python", "pytest", "pip"}
    blocked_tokens = {
        "cmd",
        "powershell",
        "bash",
        "sh",
        "curl",
        "wget",
        "rm",
        "del",
        "git",
        "npm",
        "docker",
        "&",
        "|",
        ";",
        ">",
        "<",
    }

    def __init__(
        self,
        workspace: WorkspaceManager,
        timeout_seconds: float = 10.0,
        max_output_chars: int = 8000,
    ) -> None:
        self.workspace = workspace
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars

    def run_python(self, path: str | Path) -> CommandExecutionResult:
        """Run a Python file inside the workspace."""
        try:
            safe_path = self.workspace.resolve_safe_path(path)
        except WorkspaceSecurityError as error:
            raise CommandToolError(str(error)) from error
        if safe_path.suffix != ".py":
            raise CommandToolError("Only .py files can be executed with python.")
        relative_path = self.workspace.relative_to_workspace(
            safe_path.relative_to(self.workspace.root)
        )
        return self.run_command("python", [relative_path])

    def run_pytest(self) -> CommandExecutionResult:
        """Run pytest in the workspace."""
        return self.run_command("pytest", [])

    def run_pip_check(self) -> CommandExecutionResult:
        """Run pip check in the workspace."""
        return self.run_command("pip", ["check"])

    def run_command(self, command_id: str, args: list[str] | None = None) -> CommandExecutionResult:
        """Run a whitelisted command with validated arguments."""
        args = args or []
        self._validate_command(command_id, args)
        command = self._command_vector(command_id, args)
        started_at = perf_counter()
        try:
            completed = subprocess.run(
                command,
                cwd=self.workspace.root,
                env=self._execution_env(),
                shell=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            duration = perf_counter() - started_at
            stdout = self._truncate(completed.stdout)
            stderr = self._truncate(completed.stderr)
            return CommandExecutionResult(
                success=completed.returncode == 0,
                exit_code=completed.returncode,
                stdout=stdout,
                stderr=stderr,
                duration=duration,
            )
        except subprocess.TimeoutExpired as error:
            duration = perf_counter() - started_at
            stdout = self._truncate(self._decode_output(error.stdout))
            stderr = self._truncate(self._decode_output(error.stderr) or "Command timed out.")
            return CommandExecutionResult(
                success=False,
                exit_code=-1,
                stdout=stdout,
                stderr=stderr,
                duration=duration,
                timed_out=True,
            )

    def _validate_command(self, command_id: str, args: list[str]) -> None:
        if command_id not in self.allowed_commands:
            raise CommandToolError(f"Command is not whitelisted: {command_id}")
        if command_id == "pip" and args != ["check"]:
            raise CommandToolError("Only 'pip check' is allowed.")
        if command_id == "pytest" and args:
            raise CommandToolError("pytest arguments are not allowed.")
        if command_id == "python" and len(args) != 1:
            raise CommandToolError("python requires exactly one workspace .py path.")
        for arg in args:
            self._validate_arg(arg)
        if command_id == "python" and Path(args[0]).suffix != ".py":
            raise CommandToolError("python can only run .py files.")

    def _validate_arg(self, arg: str) -> None:
        lowered = arg.lower()
        if any(token == lowered or token in lowered for token in self.blocked_tokens):
            raise CommandToolError(f"Suspicious argument is not allowed: {arg}")
        if arg.startswith("-"):
            raise CommandToolError("Command flags are not allowed.")
        try:
            self.workspace.validate_path(arg)
        except WorkspaceSecurityError as error:
            raise CommandToolError(str(error)) from error

    def _command_vector(self, command_id: str, args: list[str]) -> list[str]:
        if command_id == "python":
            return ["python", *args]
        if command_id == "pytest":
            return ["python", "-m", "pytest"]
        if command_id == "pip":
            return ["python", "-m", "pip", "check"]
        raise CommandToolError(f"Command is not whitelisted: {command_id}")

    def _execution_env(self) -> dict[str, str]:
        """Return an execution environment that can import local workspace modules."""
        env = os.environ.copy()
        workspace_path = str(self.workspace.root)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            workspace_path if not existing else f"{workspace_path}{os.pathsep}{existing}"
        )
        return env

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_output_chars:
            return text
        return text[: self.max_output_chars - 15].rstrip() + "\n...[truncated]"

    def _decode_output(self, output: str | bytes | None) -> str:
        if output is None:
            return ""
        if isinstance(output, bytes):
            return output.decode("utf-8", errors="replace")
        return output
