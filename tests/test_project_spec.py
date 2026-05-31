from pathlib import Path

from multi_agent_lab.agents.file_agent import FileAgent
from multi_agent_lab.agents.planner_agent import PlannerAgent
from multi_agent_lab.agents.spec_agent import SpecAgent
from multi_agent_lab.core.message import EventType, Message
from multi_agent_lab.core.message_bus import MessageBus
from multi_agent_lab.core.project_spec import ProjectSpec, sales_3d_printing_spec
from multi_agent_lab.core.task_graph_store import TaskGraphStore
from multi_agent_lab.core.workspace_manager import WorkspaceManager
from multi_agent_lab.runtime import AgentRuntime
from multi_agent_lab.tools.file_tool import FileTool


def test_sales_goal_project_spec_has_expected_entities() -> None:
    spec = sales_3d_printing_spec()

    assert spec.app_name == "3D Print Sales Tracker"
    assert spec.entities == ["Product", "Customer", "Sale", "SaleItem", "Payment"]
    assert "registrar venta" in spec.features
    assert "payment status required" in spec.validations


def test_project_spec_roundtrip() -> None:
    spec = sales_3d_printing_spec()

    loaded = ProjectSpec.from_dict(spec.to_dict())

    assert loaded == spec
    assert '"app_name": "3D Print Sales Tracker"' in spec.to_json()


async def test_spec_agent_generates_valid_project_spec() -> None:
    bus = MessageBus()
    generated = await bus.subscribe(EventType.SPEC_GENERATED)
    approved = await bus.subscribe(EventType.SPEC_APPROVED)
    agent = SpecAgent("spec_agent", bus)

    await agent.handle_message(
        Message(
            sender="runtime",
            type=EventType.GOAL_SUBMITTED,
            content={"goal": "Hazme una web para registrar mis ventas de impresion 3D"},
            metadata={"require_spec": True},
        )
    )

    generated_event = await generated.get()
    approved_event = await approved.get()
    spec = generated_event.content["project_spec"]
    assert spec["entities"] == ["Product", "Customer", "Sale", "SaleItem", "Payment"]
    assert approved_event.content["project_spec"] == spec


async def test_project_spec_is_saved_in_workspace(tmp_path: Path) -> None:
    bus = MessageBus()
    workspace = WorkspaceManager(tmp_path / "workspace")
    file_tool = FileTool(workspace)
    file_agent = FileAgent("file_agent", bus, file_tool, TaskGraphStore())
    spec_agent = SpecAgent("spec_agent", bus)
    written = await bus.subscribe(EventType.FILE_WRITTEN)

    await file_agent.start()
    try:
        await spec_agent.handle_message(
            Message(
                sender="runtime",
                type=EventType.GOAL_SUBMITTED,
                content={"goal": "Hazme una web para registrar mis ventas de impresion 3D"},
                metadata={"require_spec": True},
            )
        )
        event = await written.get()
    finally:
        await file_agent.stop()

    assert event.content["path"] == ".spec/project_spec.json"
    saved = file_tool.read_file(".spec/project_spec.json")
    assert '"SaleItem"' in saved


async def test_planner_receives_project_spec() -> None:
    bus = MessageBus()
    graph_store = TaskGraphStore()
    planner = PlannerAgent("planner", bus, graph_store)
    decomposed = await bus.subscribe(EventType.GOAL_DECOMPOSED)
    spec = sales_3d_printing_spec()

    await planner.handle_message(
        Message(
            sender="spec_agent",
            type=EventType.SPEC_APPROVED,
            content={
                "goal": "Hazme una web para registrar mis ventas de impresion 3D",
                "path": "README.md",
                "project_spec": spec.to_dict(),
            },
            metadata={"allow_execution": False},
        )
    )

    event = await decomposed.get()
    assert event.content["project_spec"]["app_name"] == "3D Print Sales Tracker"
    assert event.content["goal"].startswith("3D Print Sales Tracker:")


async def test_runtime_summary_marks_spec_generated(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    runtime = AgentRuntime(
        "Hazme una web para registrar mis ventas de impresion 3D",
        workspace_path=str(workspace),
        database_url=f"sqlite:///{tmp_path / 'runtime.db'}",
        use_mock_llm=True,
        timeout_seconds=5,
    )

    summary = await runtime.run()

    assert summary.status == "completed"
    assert summary.spec_generated is True
    assert (workspace / ".spec" / "project_spec.json").exists()
    assert ".spec/project_spec.json" not in summary.files_created
