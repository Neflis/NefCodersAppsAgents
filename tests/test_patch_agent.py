from pathlib import Path

from multi_agent_lab.agents.file_agent import FileAgent
from multi_agent_lab.core.message import EventType, Message
from multi_agent_lab.core.message_bus import MessageBus
from multi_agent_lab.core.task_graph_store import TaskGraphStore
from multi_agent_lab.core.workspace_manager import WorkspaceManager
from multi_agent_lab.tools.file_tool import FileTool


async def test_file_agent_publishes_patch_applied(tmp_path: Path) -> None:
    workspace = WorkspaceManager(tmp_path / "workspace")
    file_tool = FileTool(workspace)
    file_tool.write_file("README.md", "hello world\n")
    bus = MessageBus()
    applied = await bus.subscribe(EventType.PATCH_APPLIED)
    agent = FileAgent("file_agent", bus, file_tool, TaskGraphStore())

    await agent.handle_message(
        Message(
            sender="coder",
            type=EventType.PATCH_PROPOSED,
            content={
                "task_id": "patch-task",
                "path": "README.md",
                "search": "hello",
                "replace": "hola",
            },
            correlation_id="corr",
        )
    )

    event = await applied.get()
    assert event.content["path"] == "README.md"
    assert event.content["diff_summary"] == "README.md: +1 -1"
    assert file_tool.read_file("README.md") == "hola world\n"


async def test_file_agent_publishes_patch_failed(tmp_path: Path) -> None:
    workspace = WorkspaceManager(tmp_path / "workspace")
    file_tool = FileTool(workspace)
    file_tool.write_file("README.md", "hello world\n")
    bus = MessageBus()
    failed = await bus.subscribe(EventType.PATCH_FAILED)
    agent = FileAgent("file_agent", bus, file_tool, TaskGraphStore())

    await agent.handle_message(
        Message(
            sender="coder",
            type=EventType.PATCH_PROPOSED,
            content={
                "task_id": "patch-task",
                "path": "README.md",
                "search": "missing",
                "replace": "hola",
            },
            correlation_id="corr",
        )
    )

    event = await failed.get()
    assert event.content["path"] == "README.md"
    assert "not found" in event.content["error"]
    assert file_tool.read_file("README.md") == "hello world\n"
