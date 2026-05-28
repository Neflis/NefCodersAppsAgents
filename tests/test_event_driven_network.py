import inspect
from pathlib import Path

from multi_agent_lab.agents.base_agent import BaseAgent
from multi_agent_lab.agents.coder_agent import CoderAgent
from multi_agent_lab.agents.file_agent import FileAgent
from multi_agent_lab.agents.planner_agent import PlannerAgent
from multi_agent_lab.agents.reviewer_agent import ReviewerAgent
from multi_agent_lab.agents.supervisor_agent import SupervisorAgent
from multi_agent_lab.agents.task_coordinator_agent import TaskCoordinatorAgent
from multi_agent_lab.agents.tester_agent import TesterAgent as ValidationAgent
from multi_agent_lab.core.message import EventType, Message
from multi_agent_lab.core.message_bus import MessageBus
from multi_agent_lab.core.task_graph_store import TaskGraphStore
from multi_agent_lab.core.workspace_manager import WorkspaceManager
from multi_agent_lab.tools.file_tool import FileTool


async def test_supervisor_receives_all_events() -> None:
    bus = MessageBus()
    supervisor_inbox = await bus.subscribe("*")
    await bus.publish(Message(sender="demo", type=EventType.GOAL_SUBMITTED, content={}))
    await bus.publish(Message(sender="planner", type=EventType.TASK_CREATED, content={}))

    assert (await supervisor_inbox.get()).type == EventType.GOAL_SUBMITTED
    assert (await supervisor_inbox.get()).type == EventType.TASK_CREATED


def test_agents_do_not_call_each_other_directly() -> None:
    agent_classes = [
        PlannerAgent,
        CoderAgent,
        ReviewerAgent,
        FileAgent,
        ValidationAgent,
        TaskCoordinatorAgent,
    ]

    for agent_class in agent_classes:
        source = inspect.getsource(agent_class.handle_message)
        assert ".handle_message(" not in source
        assert "PlannerAgent(" not in source
        assert "CoderAgent(" not in source
        assert "ReviewerAgent(" not in source
        assert "FileAgent(" not in source
        assert "TesterAgent(" not in source


async def test_complete_event_driven_demo_writes_file(tmp_path: Path) -> None:
    bus = MessageBus()
    graph_store = TaskGraphStore()
    file_tool = FileTool(WorkspaceManager(tmp_path / "workspace"))
    test_results = await bus.subscribe(EventType.TEST_PASSED)
    agents: list[BaseAgent] = [
        PlannerAgent("planner", bus, graph_store),
        CoderAgent("coder", bus),
        ReviewerAgent("reviewer", bus, graph_store),
        FileAgent("file_agent", bus, file_tool, graph_store),
        ValidationAgent("tester", bus, file_tool),
        TaskCoordinatorAgent("coordinator", bus, graph_store),
        SupervisorAgent("supervisor", bus),
    ]

    try:
        for agent in agents:
            await agent.start()

        await bus.publish(
            Message(
                sender="test",
                type=EventType.GOAL_SUBMITTED,
                content={
                    "goal": "Crear una pequena documentacion README para una app TODO",
                    "path": "README.md",
                },
            )
        )
        result = await test_results.get()

        assert result.type == EventType.TEST_PASSED
        assert result.correlation_id is not None
        assert file_tool.exists("README.md")
        assert "TODO App" in file_tool.read_file("README.md")
    finally:
        for agent in agents:
            await agent.stop()
