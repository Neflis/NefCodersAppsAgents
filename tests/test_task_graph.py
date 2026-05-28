from multi_agent_lab.agents.coder_agent import CoderAgent
from multi_agent_lab.agents.supervisor_agent import SupervisorAgent
from multi_agent_lab.agents.task_coordinator_agent import TaskCoordinatorAgent
from multi_agent_lab.core.capability import Capability
from multi_agent_lab.core.message import EventType, Message
from multi_agent_lab.core.message_bus import MessageBus
from multi_agent_lab.core.task_graph import Goal, TaskGraph, TaskNode, TaskNodeStatus
from multi_agent_lab.core.task_graph_store import TaskGraphStore


def test_task_graph_creates_tasks_and_dependencies() -> None:
    graph = TaskGraph(Goal("demo", "corr-1"))
    first = graph.add_task(TaskNode("draft", "draft", Capability.CODING.value))
    second = graph.add_task(TaskNode("review", "review", Capability.REVIEWING.value))

    graph.add_dependency(second.id, first.id)

    assert second.id in graph.nodes
    assert graph.nodes[second.id].dependencies == {first.id}


def test_task_with_dependencies_is_not_ready_until_dependencies_complete() -> None:
    graph = TaskGraph(Goal("demo", "corr-1"))
    first = graph.add_task(TaskNode("draft", "draft", Capability.CODING.value))
    second = graph.add_task(TaskNode("review", "review", Capability.REVIEWING.value))
    graph.add_dependency(second.id, first.id)

    ready = graph.ready_tasks()
    assert [task.id for task in ready] == [first.id]
    assert graph.nodes[second.id].status == TaskNodeStatus.PENDING

    graph.complete_task(first.id, {"content": "ok"})
    ready = graph.ready_tasks()

    assert [task.id for task in ready] == [second.id]


def test_agent_only_claims_compatible_capability() -> None:
    bus = MessageBus()
    coder = CoderAgent("coder", bus)
    compatible = Message(
        sender="planner",
        type=EventType.TASK_READY,
        content={
            "task_id": "1",
            "required_capability": Capability.CODING.value,
            "status": TaskNodeStatus.READY.value,
        },
    )
    incompatible = Message(
        sender="planner",
        type=EventType.TASK_READY,
        content={
            "task_id": "2",
            "required_capability": Capability.REVIEWING.value,
            "status": TaskNodeStatus.READY.value,
        },
    )

    assert coder.can_claim(compatible)
    assert not coder.can_claim(incompatible)


async def test_task_coordinator_releases_dependent_tasks() -> None:
    bus = MessageBus()
    graph_store = TaskGraphStore()
    graph = TaskGraph(Goal("demo", "corr-1"))
    first = graph.add_task(TaskNode("draft", "draft", Capability.CODING.value))
    second = graph.add_task(TaskNode("review", "review", Capability.REVIEWING.value))
    graph.add_dependency(second.id, first.id)
    graph.ready_tasks()
    graph_store.add(graph)
    ready_inbox = await bus.subscribe(EventType.TASK_READY)
    coordinator = TaskCoordinatorAgent("coordinator", bus, graph_store)
    await coordinator.start()

    try:
        await bus.publish(
            Message(
                sender="coder",
                type=EventType.TASK_COMPLETED,
                content={"task_id": first.id, "result": {"content": "ok"}},
                correlation_id="corr-1",
            )
        )
        ready = await ready_inbox.get()
        assert ready.content["task_id"] == second.id
    finally:
        await coordinator.stop()


async def test_task_coordinator_retries_failed_task() -> None:
    bus = MessageBus()
    graph_store = TaskGraphStore()
    graph = TaskGraph(Goal("demo", "corr-1"))
    task = graph.add_task(TaskNode("draft", "draft", Capability.CODING.value))
    graph.ready_tasks()
    graph_store.add(graph)
    retried_inbox = await bus.subscribe(EventType.TASK_RETRIED)
    coordinator = TaskCoordinatorAgent("coordinator", bus, graph_store)
    await coordinator.start()

    try:
        await bus.publish(
            Message(
                sender="coder",
                type=EventType.TASK_FAILED,
                content={"task_id": task.id, "error": "temporary"},
                correlation_id="corr-1",
            )
        )
        retried = await retried_inbox.get()
        assert retried.content["task_id"] == task.id
        assert graph.nodes[task.id].retries == 1
    finally:
        await coordinator.stop()


async def test_supervisor_detects_excessive_retries() -> None:
    bus = MessageBus()
    halted_inbox = await bus.subscribe(EventType.WORKFLOW_HALTED)
    supervisor = SupervisorAgent("supervisor", bus, max_retries_per_correlation=1)
    await supervisor.start()

    try:
        for retry in range(2):
            await bus.publish(
                Message(
                    sender="coordinator",
                    type=EventType.TASK_RETRIED,
                    content={"task_id": "1", "retries": retry + 1},
                    correlation_id="corr-1",
                )
            )
        halted = await halted_inbox.get()
        assert halted.content["reason"] == "retry_limit"
    finally:
        await supervisor.stop()
