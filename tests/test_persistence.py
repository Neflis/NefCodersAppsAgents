from multi_agent_lab.core.agent_event_logger import AgentEventLogger
from multi_agent_lab.core.message import Message
from multi_agent_lab.core.message_bus import MessageBus
from multi_agent_lab.core.sqlite_store import SQLiteStore
from multi_agent_lab.core.task import Task, TaskStatus
from multi_agent_lab.core.task_queue import TaskQueue


def make_store(tmp_path) -> SQLiteStore:
    return SQLiteStore(f"sqlite:///{tmp_path / 'test.db'}")


async def test_message_bus_persists_published_messages(tmp_path) -> None:
    store = make_store(tmp_path)
    bus = MessageBus(store)
    message = Message(
        sender="planner", receiver="coder", type="task.created", content={"task_id": "1"}
    )

    await bus.publish(message)

    rows = store.fetch_all("messages")
    assert len(rows) == 1
    assert rows[0]["id"] == message.id
    assert rows[0]["sender"] == "planner"
    store.close()


async def test_task_queue_persists_tasks_and_status_changes(tmp_path) -> None:
    store = make_store(tmp_path)
    queue = TaskQueue(store)
    task = Task(title="demo", description="demo task")

    await queue.put(task)
    await queue.update_status(task, TaskStatus.DONE)

    rows = store.fetch_all("tasks")
    assert len(rows) == 1
    assert rows[0]["id"] == task.id
    assert rows[0]["status"] == "done"
    store.close()


async def test_agent_event_logger_persists_events(tmp_path) -> None:
    store = make_store(tmp_path)
    logger = AgentEventLogger(store)

    await logger.log("planner", "agent_started", {"demo": True})

    rows = store.fetch_all("agent_events")
    assert len(rows) == 1
    assert rows[0]["agent"] == "planner"
    assert rows[0]["event_type"] == "agent_started"
    store.close()
