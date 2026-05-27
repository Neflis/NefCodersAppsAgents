"""Priority-aware async task queue."""

from __future__ import annotations

import asyncio
from itertools import count

from multi_agent_lab.core.sqlite_store import SQLiteStore
from multi_agent_lab.core.task import Task, TaskStatus


class TaskQueue:
    """Priority-aware task queue with optional persistence."""

    def __init__(self, store: SQLiteStore | None = None) -> None:
        self._queue: asyncio.PriorityQueue[tuple[int, int, Task]] = asyncio.PriorityQueue()
        self._sequence = count()
        self._store = store

    async def put(self, task: Task) -> None:
        """Add a task to the queue."""
        if self._store is not None:
            self._store.save_task(task)
        await self._queue.put((-task.priority, next(self._sequence), task))

    async def get(self) -> Task:
        """Return the next task by priority."""
        _, _, task = await self._queue.get()
        return task

    async def update_status(self, task: Task, status: TaskStatus) -> None:
        """Update task status in memory and persistence."""
        task.status = status
        if self._store is not None:
            self._store.update_task_status(task.id, status)

    def task_done(self) -> None:
        """Mark a previously fetched task as complete for queue accounting."""
        self._queue.task_done()

    async def join(self) -> None:
        """Wait until all queued tasks have been marked complete."""
        await self._queue.join()

    def empty(self) -> bool:
        """Return whether the queue has no pending tasks."""
        return self._queue.empty()

    def qsize(self) -> int:
        """Return the number of queued tasks."""
        return self._queue.qsize()
