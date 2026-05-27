from multi_agent_lab.core.task import Task
from multi_agent_lab.core.task_queue import TaskQueue


async def test_task_queue_returns_highest_priority_first() -> None:
    queue = TaskQueue()
    low = Task(title="low", description="low priority", priority=1)
    high = Task(title="high", description="high priority", priority=5)

    await queue.put(low)
    await queue.put(high)

    assert await queue.get() == high
    assert await queue.get() == low


async def test_task_queue_tracks_size() -> None:
    queue = TaskQueue()
    task = Task(title="task", description="demo")

    await queue.put(task)

    assert queue.qsize() == 1
    assert not queue.empty()
