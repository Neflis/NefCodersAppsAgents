import pytest

from multi_agent_lab.agents.coder_agent import CoderAgent
from multi_agent_lab.agents.planner_agent import PlannerAgent
from multi_agent_lab.agents.reviewer_agent import ReviewerAgent
from multi_agent_lab.core.capability import Capability
from multi_agent_lab.core.message import EventType, Message
from multi_agent_lab.core.message_bus import MessageBus
from multi_agent_lab.core.task_graph import Goal, TaskGraph, TaskNode
from multi_agent_lab.core.task_graph_store import TaskGraphStore
from multi_agent_lab.llm.ollama_client import InvalidJSONError, OllamaClient


async def test_mock_llm_returns_valid_json() -> None:
    client = OllamaClient(use_mock=True)

    data = await client.generate_json("CoderAgent")

    assert data["action"] == "generate_content"
    assert "content" in data


async def test_invalid_json_is_reported() -> None:
    client = OllamaClient(use_mock=True, mock_responses=["not-json"], retries=0)  # type: ignore[list-item]

    with pytest.raises(InvalidJSONError):
        await client.generate_json("bad")


async def test_planner_creates_tasks_from_llm_decision() -> None:
    bus = MessageBus()
    graph_store = TaskGraphStore()
    task_ready = await bus.subscribe(EventType.TASK_READY)
    client = OllamaClient(
        use_mock=True,
        mock_responses=[
            {
                "action": "decompose_goal",
                "reasoning_summary": "custom",
                "confidence": 1,
                "content": [
                    {
                        "title": "Custom draft",
                        "description": "draft",
                        "required_capability": Capability.CODING.value,
                        "payload": {"path": "README.md"},
                        "priority": 5,
                    }
                ],
            }
        ],
    )
    planner = PlannerAgent("planner", bus, graph_store, llm_client=client)

    await planner.handle_message(
        Message(
            sender="test",
            type=EventType.GOAL_SUBMITTED,
            content={"goal": "demo", "path": "README.md"},
        )
    )

    ready = await task_ready.get()
    assert ready.content["title"] == "Custom draft"


async def test_coder_generates_content_from_llm_decision() -> None:
    bus = MessageBus()
    proposed = await bus.subscribe(EventType.CODE_PROPOSED)
    client = OllamaClient(
        use_mock=True,
        mock_responses=[
            {
                "action": "generate_content",
                "reasoning_summary": "custom",
                "confidence": 1,
                "content": "# Custom README",
            }
        ],
    )
    coder = CoderAgent("coder", bus, ollama_client=client)

    await coder.handle_message(
        Message(
            sender="planner",
            type=EventType.TASK_READY,
            content={
                "task_id": "task-1",
                "title": "draft",
                "required_capability": Capability.CODING.value,
                "status": "READY",
                "payload": {"path": "README.md"},
            },
        )
    )

    event = await proposed.get()
    assert event.content["content"] == "# Custom README"


async def test_reviewer_approves_from_llm_decision() -> None:
    bus = MessageBus()
    approved = await bus.subscribe(EventType.REVIEW_APPROVED)
    graph_store = TaskGraphStore()
    graph = TaskGraph(Goal("demo", "corr"))
    draft = graph.add_task(TaskNode("draft", "draft", Capability.CODING.value))
    review = graph.add_task(TaskNode("review", "review", Capability.REVIEWING.value))
    graph.add_dependency(review.id, draft.id)
    graph.complete_task(draft.id, {"path": "README.md", "content": "# OK"})
    graph.ready_tasks()
    graph_store.add(graph)
    client = OllamaClient(
        use_mock=True,
        mock_responses=[
            {
                "action": "approve",
                "reasoning_summary": "ok",
                "confidence": 1,
                "content": {"approved": True},
            }
        ],
    )
    reviewer = ReviewerAgent("reviewer", bus, graph_store, llm_client=client)

    await reviewer.handle_message(
        Message(
            sender="coordinator",
            type=EventType.TASK_READY,
            content={
                "task_id": review.id,
                "title": "review",
                "required_capability": Capability.REVIEWING.value,
                "status": "READY",
                "payload": {"path": "README.md"},
            },
            correlation_id="corr",
        )
    )

    event = await approved.get()
    assert event.content["approved"] is True
