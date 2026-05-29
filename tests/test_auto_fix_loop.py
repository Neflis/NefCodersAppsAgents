import asyncio
from pathlib import Path

from multi_agent_lab.agents.coder_agent import CoderAgent
from multi_agent_lab.agents.file_agent import FileAgent
from multi_agent_lab.agents.supervisor_agent import SupervisorAgent
from multi_agent_lab.agents.task_coordinator_agent import TaskCoordinatorAgent
from multi_agent_lab.agents.tester_execution_agent import (
    TesterExecutionAgent as ExecutionAgent,
)
from multi_agent_lab.core.capability import Capability
from multi_agent_lab.core.message import EventType, Message
from multi_agent_lab.core.message_bus import MessageBus
from multi_agent_lab.core.task_graph import Goal, TaskGraph, TaskNode, TaskNodeStatus
from multi_agent_lab.core.task_graph_store import TaskGraphStore
from multi_agent_lab.core.workspace_manager import WorkspaceManager
from multi_agent_lab.runtime import AgentRuntime
from multi_agent_lab.tools.command_tool import CommandTool
from multi_agent_lab.tools.file_tool import FileTool


def execution_failure(task_id: str = "exec-task") -> Message:
    return Message(
        sender="tester_execution",
        type=EventType.TEST_EXECUTION_FAILED,
        content={
            "task_id": task_id,
            "command": "pytest",
            "command_id": "pytest",
            "args": [],
            "exit_code": 1,
            "stdout": "FAILED tests/test_app.py::test_readme",
            "stderr": "AssertionError: README.md debe mencionar /todos",
            "failed_files": ["tests/test_app.py"],
            "suggested_focus_files": ["README.md"],
        },
        correlation_id="corr",
    )


def graph_with_execution_task() -> tuple[TaskGraphStore, str]:
    graph_store = TaskGraphStore()
    graph = TaskGraph(Goal("demo", "corr"))
    execution = graph.add_task(
        TaskNode(
            title="Ejecutar validacion pytest",
            description="pytest",
            required_capability=Capability.TESTING_EXECUTION.value,
            payload={"command_id": "pytest", "args": []},
            status=TaskNodeStatus.IN_PROGRESS,
            id="exec-task",
        )
    )
    graph_store.add(graph)
    return graph_store, execution.id


async def test_pytest_failure_generates_fix_requested() -> None:
    bus = MessageBus()
    graph_store, task_id = graph_with_execution_task()
    coordinator = TaskCoordinatorAgent("coordinator", bus, graph_store)
    requested = await bus.subscribe(EventType.FIX_REQUESTED)
    task_ready = await bus.subscribe(EventType.TASK_READY)

    await coordinator.handle_message(execution_failure(task_id))

    event = await requested.get()
    ready = await task_ready.get()
    assert event.content["required_capability"] == Capability.CODING.value
    assert event.content["payload"]["type"] == "fix"
    assert event.content["payload"]["path"] == "README.md"
    assert event.metadata["failed_command"] == "pytest"
    assert event.metadata["fix_attempt"] == 1
    assert event.metadata["failure_context"]["failure_type"] == "AssertionError"
    assert ready.content["payload"]["type"] == "fix"
    assert (
        ready.content["payload"]["failure_context"]["failing_test"]
        == "tests/test_app.py::test_readme"
    )


async def test_coordinator_filters_external_failure_paths(tmp_path: Path) -> None:
    bus = MessageBus()
    graph_store, task_id = graph_with_execution_task()
    coordinator = TaskCoordinatorAgent(
        "coordinator",
        bus,
        graph_store,
        path_normalizer=None,
    )
    requested = await bus.subscribe(EventType.FIX_REQUESTED)
    failure = execution_failure(task_id)
    failure.content["suggested_focus_files"] = [str(tmp_path / "outside.py"), "README.md"]
    failure.content["stderr"] = (
        f'Traceback (most recent call last):\n  File "{tmp_path / "outside.py"}", line 3\n'
        "AssertionError"
    )

    await coordinator.handle_message(failure)

    event = await requested.get()
    assert event.content["payload"]["suggested_focus_files"] == ["README.md"]
    suspected_files = event.content["payload"]["failure_context"]["suspected_files"]
    assert suspected_files == ["tests/test_app.py"]
    assert str(tmp_path / "outside.py") not in suspected_files


async def test_coder_proposes_fix_using_stderr() -> None:
    bus = MessageBus()
    proposed = await bus.subscribe(EventType.FIX_PROPOSED)
    coder = CoderAgent("coder", bus)
    message = Message(
        sender="coordinator",
        type=EventType.FIX_REQUESTED,
        content={
            "task_id": "fix-task",
            "required_capability": Capability.CODING.value,
            "status": "READY",
            "payload": {
                "type": "fix",
                "path": "README.md",
                "failure": execution_failure().content,
                "suggested_focus_files": ["README.md"],
            },
        },
        correlation_id="corr",
    )

    await coder.handle_message(message)

    event = await proposed.get()
    assert event.content["path"] == "README.md"
    assert "/todos" in event.content["content"]
    assert "AssertionError" in event.content["based_on_error"]
    assert event.content["fix_strategy"] == "patch_existing_file"
    assert event.content["content_hash"]


async def test_file_agent_applies_fix(tmp_path: Path) -> None:
    bus = MessageBus()
    applied = await bus.subscribe(EventType.FIX_APPLIED)
    workspace = WorkspaceManager(tmp_path / "workspace")
    file_tool = FileTool(workspace)
    graph_store = TaskGraphStore()
    agent = FileAgent("file_agent", bus, file_tool, graph_store)

    await agent.handle_message(
        Message(
            sender="coder",
            type=EventType.FIX_PROPOSED,
            content={
                "task_id": "fix-task",
                "path": "README.md",
                "content": "# Flask TODO API\n\n/todos\n",
                "execution_task_id": "exec-task",
                "command_id": "pytest",
                "args": [],
            },
            correlation_id="corr",
        )
    )

    event = await applied.get()
    assert event.content["path"] == "README.md"
    assert (tmp_path / "workspace" / "README.md").exists()


async def test_fix_applied_publishes_retest_requested(tmp_path: Path) -> None:
    bus = MessageBus()
    requested = await bus.subscribe(EventType.RETEST_REQUESTED)
    agent = ExecutionAgent("tester_execution", bus, CommandTool(WorkspaceManager(tmp_path)))

    await agent.handle_message(
        Message(
            sender="file_agent",
            type=EventType.FIX_APPLIED,
            content={
                "task_id": "fix-task",
                "execution_task_id": "exec-task",
                "command_id": "pytest",
                "args": [],
            },
            correlation_id="corr",
        )
    )

    event = await requested.get()
    assert event.content["task_id"] == "exec-task"
    assert event.content["command_id"] == "pytest"


def test_second_pytest_can_pass_after_fix(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    tests_dir = workspace / "tests"
    tests_dir.mkdir(parents=True)
    (workspace / "README.md").write_text("# Bad\n", encoding="utf-8")
    (tests_dir / "test_readme.py").write_text(
        "from pathlib import Path\n\n"
        "def test_readme_mentions_endpoint():\n"
        "    assert '/todos' in Path('README.md').read_text(encoding='utf-8')\n",
        encoding="utf-8",
    )
    tool = CommandTool(WorkspaceManager(workspace))

    first = tool.run_pytest()
    (workspace / "README.md").write_text("# Flask TODO API\n\n/todos\n", encoding="utf-8")
    second = tool.run_pytest()

    assert not first.success
    assert second.success


async def test_workflow_does_not_halt_until_fix_retries_exhausted() -> None:
    bus = MessageBus()
    graph_store, task_id = graph_with_execution_task()
    coordinator = TaskCoordinatorAgent("coordinator", bus, graph_store, max_fix_attempts=2)
    halted = await bus.subscribe(EventType.WORKFLOW_HALTED)
    requested = await bus.subscribe(EventType.FIX_REQUESTED)

    await coordinator.handle_message(execution_failure(task_id))
    await requested.get()
    await coordinator.handle_message(execution_failure(task_id))
    await requested.get()

    try:
        await asyncio.wait_for(halted.get(), timeout=0.05)
    except TimeoutError:
        pass
    else:
        raise AssertionError("Workflow halted before fix retries were exhausted.")

    await coordinator.handle_message(execution_failure(task_id))
    event = await halted.get()
    assert event.content["reason"] == "max_fix_attempts_exceeded"


async def test_event_limit_does_not_cut_active_fix_loop() -> None:
    bus = MessageBus()
    halted = await bus.subscribe(EventType.WORKFLOW_HALTED)
    supervisor = SupervisorAgent("supervisor", bus, max_events_per_correlation=1)

    await supervisor.handle_message(
        Message(
            sender="tester",
            type=EventType.TEST_EXECUTION_FAILED,
            content={},
            correlation_id="c",
        )
    )
    await supervisor.handle_message(
        Message(sender="coordinator", type=EventType.FIX_REQUESTED, content={}, correlation_id="c")
    )
    await supervisor.handle_message(
        Message(sender="coder", type=EventType.FIX_PROPOSED, content={}, correlation_id="c")
    )

    try:
        await asyncio.wait_for(halted.get(), timeout=0.05)
    except TimeoutError:
        pass
    else:
        raise AssertionError("event limit interrupted an active fix loop")


async def test_runtime_summary_excludes_pytest_cache(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    runtime = AgentRuntime(
        "Crea una pequena API Flask TODO con tests basicos",
        workspace_path=str(workspace),
        database_url=f"sqlite:///{tmp_path / 'runtime.db'}",
        use_mock_llm=True,
        timeout_seconds=10,
        allow_execution=True,
    )

    summary = await runtime.run()

    assert summary.status == "completed"
    assert ".pytest_cache" not in " ".join(summary.files_created)
    assert "__pycache__" not in " ".join(summary.files_created)
