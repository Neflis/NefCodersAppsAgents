from pathlib import Path

import pytest

from multi_agent_lab.agents.tester_execution_agent import (
    TesterExecutionAgent as ExecutionAgent,
)
from multi_agent_lab.core.capability import Capability
from multi_agent_lab.core.message import EventType, Message
from multi_agent_lab.core.message_bus import MessageBus
from multi_agent_lab.core.workspace_manager import WorkspaceManager
from multi_agent_lab.tools.command_tool import (
    CommandExecutionResult,
    CommandTool,
    CommandToolError,
)


def command_tool(tmp_path: Path, timeout: float = 2.0, max_output: int = 8000) -> CommandTool:
    return CommandTool(
        WorkspaceManager(tmp_path / "workspace"),
        timeout_seconds=timeout,
        max_output_chars=max_output,
    )


def test_pytest_is_allowed(tmp_path: Path) -> None:
    tool = command_tool(tmp_path)
    tests_dir = tmp_path / "workspace" / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")

    result = tool.run_pytest()

    assert result.success
    assert result.exit_code == 0


def test_pytest_can_import_workspace_app_module(tmp_path: Path) -> None:
    tool = command_tool(tmp_path)
    workspace = tmp_path / "workspace"
    tests_dir = workspace / "tests"
    tests_dir.mkdir(parents=True)
    (workspace / "app.py").write_text("app = object()\n", encoding="utf-8")
    (tests_dir / "test_app.py").write_text(
        "from app import app\n\n\ndef test_imports_app():\n    assert app is not None\n",
        encoding="utf-8",
    )

    result = tool.run_pytest()

    assert result.success


def test_pytest_can_target_safe_test_file(tmp_path: Path) -> None:
    tool = command_tool(tmp_path)
    workspace = tmp_path / "workspace"
    tests_dir = workspace / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test_selected.py").write_text(
        "def test_selected():\n    assert True\n",
        encoding="utf-8",
    )

    result = tool.run_pytest(["tests/test_selected.py"])

    assert result.success


def test_pytest_rejects_unsafe_target(tmp_path: Path) -> None:
    tool = command_tool(tmp_path)

    with pytest.raises(CommandToolError):
        tool.run_pytest(["../outside.py"])

    with pytest.raises(CommandToolError):
        tool.run_pytest(["task_cli.py"])


def test_pip_install_requirements_is_allowed(tmp_path: Path) -> None:
    tool = command_tool(tmp_path)

    command = tool._command_vector("pip", ["install", "-r", "requirements.txt"])

    assert command == ["python", "-m", "pip", "install", "-r", "requirements.txt"]


def test_mvn_test_is_allowed(tmp_path: Path) -> None:
    tool = command_tool(tmp_path)

    tool._validate_command("mvn", ["test"])
    command = tool._command_vector("mvn", ["test"])

    assert Path(command[0]).name.lower() in {"mvn", "mvn.cmd"}
    assert command[1:] == ["test"]


def test_mvn_blocks_non_test_goals(tmp_path: Path) -> None:
    tool = command_tool(tmp_path)

    with pytest.raises(CommandToolError):
        tool.run_command("mvn", ["clean", "install"])

    with pytest.raises(CommandToolError):
        tool.run_command("mvn", [])


def test_npm_install_and_build_are_allowed(tmp_path: Path) -> None:
    tool = command_tool(tmp_path)

    install = tool._command_vector("npm", ["install"])
    build = tool._command_vector("npm", ["run", "build"])

    assert Path(install[0]).name.lower() == "npm.cmd"
    assert install[1:] == ["install"]
    assert Path(build[0]).name.lower() == "npm.cmd"
    assert build[1:] == ["run", "build"]


def test_npm_blocks_other_commands(tmp_path: Path) -> None:
    tool = command_tool(tmp_path)

    with pytest.raises(CommandToolError):
        tool.run_command("npm", ["test"])

    with pytest.raises(CommandToolError):
        tool.run_command("npm", ["run", "start"])


def test_shell_commands_are_blocked(tmp_path: Path) -> None:
    tool = command_tool(tmp_path)

    with pytest.raises(CommandToolError):
        tool.run_command("powershell", [])

    with pytest.raises(CommandToolError):
        tool.run_command("python", [])


def test_paths_outside_workspace_are_blocked(tmp_path: Path) -> None:
    tool = command_tool(tmp_path)

    with pytest.raises(CommandToolError):
        tool.run_python("../outside.py")


def test_timeout_works(tmp_path: Path) -> None:
    tool = command_tool(tmp_path, timeout=0.1)
    (tmp_path / "workspace" / "slow.py").write_text(
        "import time\ntime.sleep(2)\n",
        encoding="utf-8",
    )

    result = tool.run_python("slow.py")

    assert not result.success
    assert result.timed_out


def test_stdout_is_truncated(tmp_path: Path) -> None:
    tool = command_tool(tmp_path, max_output=40)
    (tmp_path / "workspace" / "loud.py").write_text(
        "print('x' * 200)\n",
        encoding="utf-8",
    )

    result = tool.run_python("loud.py")

    assert result.success
    assert len(result.stdout) <= 40
    assert "[truncated]" in result.stdout


def test_minimal_flask_app_executes(tmp_path: Path) -> None:
    tool = command_tool(tmp_path)
    workspace = tmp_path / "workspace"
    (workspace / "flask.py").write_text(
        "class Flask:\n    def __init__(self, name):\n        self.name = name\n",
        encoding="utf-8",
    )
    (workspace / "app.py").write_text(
        "from flask import Flask\napp = Flask(__name__)\nprint(app.name)\n",
        encoding="utf-8",
    )

    result = tool.run_python("app.py")

    assert result.success
    assert "__main__" in result.stdout


async def test_import_failure_publishes_execution_failed(tmp_path: Path) -> None:
    bus = MessageBus()
    failed = await bus.subscribe(EventType.TEST_EXECUTION_FAILED)
    tool = command_tool(tmp_path)
    (tmp_path / "workspace" / "app.py").write_text(
        "import definitely_missing_package\n",
        encoding="utf-8",
    )
    agent = ExecutionAgent("tester_execution", bus, tool)

    await agent.handle_message(
        Message(
            sender="coordinator",
            type=EventType.TEST_EXECUTION_REQUESTED,
            content={"task_id": "task-1", "command_id": "python", "args": ["app.py"]},
            correlation_id="corr",
        )
    )

    event = await failed.get()
    assert event.content["task_id"] == "task-1"
    assert "ModuleNotFoundError" in event.content["result"]["stderr"]


async def test_execution_agent_claims_ready_task(tmp_path: Path) -> None:
    bus = MessageBus()
    requested = await bus.subscribe(EventType.TEST_EXECUTION_REQUESTED)
    agent = ExecutionAgent("tester_execution", bus, command_tool(tmp_path))

    await agent.handle_message(
        Message(
            sender="coordinator",
            type=EventType.TASK_READY,
            content={
                "task_id": "task-1",
                "required_capability": Capability.TESTING_EXECUTION.value,
                "status": "READY",
                "payload": {"command_id": "pytest", "args": []},
            },
            correlation_id="corr",
        )
    )

    event = await requested.get()
    assert event.content["command_id"] == "pytest"


async def test_execution_agent_uses_run_pytest_entrypoint() -> None:
    class FakeCommandTool:
        def __init__(self) -> None:
            self.used_run_pytest = False
            self.used_run_command = False

        def run_pytest(self, args=None):  # noqa: ANN001
            self.used_run_pytest = True
            self.args = args or []
            return type(
                "Result",
                (),
                {
                    "success": True,
                    "to_dict": lambda self: {
                        "success": True,
                        "exit_code": 0,
                        "stdout": "",
                        "stderr": "",
                    },
                },
            )()

        def run_command(self, command_id, args):  # noqa: ANN001
            self.used_run_command = True
            raise AssertionError("pytest must use run_pytest")

    bus = MessageBus()
    passed = await bus.subscribe(EventType.TEST_EXECUTION_PASSED)
    tool = FakeCommandTool()
    agent = ExecutionAgent("tester_execution", bus, tool)  # type: ignore[arg-type]

    await agent.handle_message(
        Message(
            sender="coordinator",
            type=EventType.TEST_EXECUTION_REQUESTED,
            content={"task_id": "task-1", "command_id": "pytest", "args": []},
            correlation_id="corr",
        )
    )

    await passed.get()
    assert tool.used_run_pytest
    assert tool.args == []
    assert not tool.used_run_command


async def test_execution_agent_selects_mvn_when_pom_exists(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "pom.xml").write_text("<project></project>\n", encoding="utf-8")
    bus = MessageBus()
    requested = await bus.subscribe(EventType.TEST_EXECUTION_REQUESTED)
    agent = ExecutionAgent("tester_execution", bus, command_tool(tmp_path))

    await agent.handle_message(
        Message(
            sender="coordinator",
            type=EventType.TASK_READY,
            content={
                "task_id": "task-1",
                "required_capability": Capability.TESTING_EXECUTION.value,
                "status": "READY",
                "payload": {"command_id": "pytest", "args": []},
            },
            correlation_id="corr",
        )
    )

    event = await requested.get()
    assert event.content["command_id"] == "mvn"
    assert event.content["args"] == ["test"]


async def test_execution_agent_selects_npm_install_when_package_exists(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "package.json").write_text('{"scripts":{"build":"ngc"}}\n', encoding="utf-8")
    bus = MessageBus()
    requested = await bus.subscribe(EventType.TEST_EXECUTION_REQUESTED)
    agent = ExecutionAgent("tester_execution", bus, command_tool(tmp_path))

    await agent.handle_message(
        Message(
            sender="coordinator",
            type=EventType.TASK_READY,
            content={
                "task_id": "task-1",
                "required_capability": Capability.TESTING_EXECUTION.value,
                "status": "READY",
                "payload": {"command_id": "pytest", "args": []},
            },
            correlation_id="corr",
        )
    )

    event = await requested.get()
    assert event.content["command_id"] == "npm"
    assert event.content["args"] == ["install"]


async def test_execution_agent_runs_mvn_test_through_command_tool() -> None:
    class FakeCommandTool:
        def __init__(self) -> None:
            self.used_run_command = False
            self.command_id = ""
            self.args = []

        def run_pytest(self, args=None):  # noqa: ANN001
            raise AssertionError("mvn must not use run_pytest")

        def run_command(self, command_id, args):  # noqa: ANN001
            self.used_run_command = True
            self.command_id = command_id
            self.args = args
            return CommandExecutionResult(
                success=True,
                exit_code=0,
                stdout="",
                stderr="",
                duration=0.0,
            )

    bus = MessageBus()
    passed = await bus.subscribe(EventType.TEST_EXECUTION_PASSED)
    tool = FakeCommandTool()
    agent = ExecutionAgent("tester_execution", bus, tool)  # type: ignore[arg-type]

    await agent.handle_message(
        Message(
            sender="coordinator",
            type=EventType.TEST_EXECUTION_REQUESTED,
            content={"task_id": "task-1", "command_id": "mvn", "args": ["test"]},
            correlation_id="corr",
        )
    )

    await passed.get()
    assert tool.used_run_command
    assert tool.command_id == "mvn"
    assert tool.args == ["test"]


async def test_execution_agent_publishes_build_events_for_npm_build() -> None:
    class FakeCommandTool:
        def run_command(self, command_id, args):  # noqa: ANN001
            assert command_id == "npm"
            assert args == ["run", "build"]
            return CommandExecutionResult(
                success=True,
                exit_code=0,
                stdout="built",
                stderr="",
                duration=0.0,
            )

    bus = MessageBus()
    started = await bus.subscribe(EventType.BUILD_STARTED)
    passed = await bus.subscribe(EventType.BUILD_PASSED)
    agent = ExecutionAgent("tester_execution", bus, FakeCommandTool())  # type: ignore[arg-type]

    await agent.handle_message(
        Message(
            sender="coordinator",
            type=EventType.TEST_EXECUTION_REQUESTED,
            content={"task_id": "task-1", "command_id": "npm", "args": ["run", "build"]},
            correlation_id="corr",
        )
    )

    assert (await started.get()).content["command_id"] == "npm"
    assert (await passed.get()).content["stdout"] == "built"
