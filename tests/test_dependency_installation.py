from dataclasses import dataclass

from multi_agent_lab.agents.dependency_installer_agent import DependencyInstallerAgent
from multi_agent_lab.agents.file_agent import FileAgent
from multi_agent_lab.core.failure_analysis import FixStrategy
from multi_agent_lab.core.message import EventType, Message
from multi_agent_lab.core.message_bus import MessageBus
from multi_agent_lab.core.task_graph_store import TaskGraphStore
from multi_agent_lab.core.workspace_manager import WorkspaceManager
from multi_agent_lab.tools.command_tool import CommandExecutionResult
from multi_agent_lab.tools.file_tool import FileTool


@dataclass
class FakeInstallCommandTool:
    success: bool = True
    calls: int = 0

    def run_pip_install_requirements(self) -> CommandExecutionResult:
        self.calls += 1
        return CommandExecutionResult(
            success=self.success,
            exit_code=0 if self.success else 1,
            stdout="installed",
            stderr="" if self.success else "install failed",
            duration=0.01,
        )


async def test_requirements_fix_requests_dependency_install(tmp_path) -> None:
    bus = MessageBus()
    requested = await bus.subscribe(EventType.DEPENDENCY_INSTALL_REQUESTED)
    workspace = WorkspaceManager(tmp_path / "workspace")
    agent = FileAgent("file_agent", bus, FileTool(workspace), TaskGraphStore())

    await agent.handle_message(
        Message(
            sender="coder",
            type=EventType.FIX_PROPOSED,
            content={
                "task_id": "fix-task",
                "path": "requirements.txt",
                "content": "Flask>=3.0\n",
                "fix_strategy": FixStrategy.ADD_MISSING_DEPENDENCY.value,
                "execution_task_id": "exec-task",
                "command_id": "pytest",
                "args": [],
            },
            correlation_id="corr",
        )
    )

    event = await requested.get()
    assert event.content["requirements_path"] == "requirements.txt"
    assert event.content["execution_task_id"] == "exec-task"


async def test_dependency_installer_publishes_success_and_retest() -> None:
    bus = MessageBus()
    succeeded = await bus.subscribe(EventType.DEPENDENCY_INSTALL_SUCCEEDED)
    retest = await bus.subscribe(EventType.RETEST_REQUESTED)
    tool = FakeInstallCommandTool(success=True)
    agent = DependencyInstallerAgent("dependency_installer", bus, tool)  # type: ignore[arg-type]

    await agent.handle_message(
        Message(
            sender="file_agent",
            type=EventType.DEPENDENCY_INSTALL_REQUESTED,
            content={
                "task_id": "fix-task",
                "execution_task_id": "exec-task",
                "command_id": "pytest",
                "args": [],
            },
            correlation_id="corr",
        )
    )

    success_event = await succeeded.get()
    retest_event = await retest.get()
    assert tool.calls == 1
    assert success_event.content["result"]["success"]
    assert retest_event.content["task_id"] == "exec-task"
    assert retest_event.content["command_id"] == "pytest"


async def test_dependency_installer_publishes_failure() -> None:
    bus = MessageBus()
    failed = await bus.subscribe(EventType.DEPENDENCY_INSTALL_FAILED)
    tool = FakeInstallCommandTool(success=False)
    agent = DependencyInstallerAgent("dependency_installer", bus, tool)  # type: ignore[arg-type]

    await agent.handle_message(
        Message(
            sender="file_agent",
            type=EventType.DEPENDENCY_INSTALL_REQUESTED,
            content={"task_id": "fix-task", "execution_task_id": "exec-task"},
            correlation_id="corr",
        )
    )

    event = await failed.get()
    assert event.content["result"]["success"] is False
