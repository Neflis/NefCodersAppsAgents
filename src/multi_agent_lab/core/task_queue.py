"""Priority-aware async task queue."""

from __future__ import annotations

import asyncio
from itertools import count

from multi_agent_lab.core.task import Task
from multi_agent_lab.core.task import TaskStatus
from multi_agent_lab.core.sqlite_store import SQLiteStore


class TaskQueue:
    def __init__(self, store: SQLiteStore | None = None) -> None:
        self._queue: asyncio.PriorityQueue[tuple[int, int, Task]] = asyncio.PriorityQueue()
        self._sequence = count()
        self._store = store

    async def put(self, task: Task) -> None:
        if self._store is not None:
            self._store.save_task(task)
        await self._queue.put((-task.priority, next(self._sequence), task))

    async def get(self) -> Task:
        _, _, task = await self._queue.get()
        return task

    async def update_status(self, task: Task, status: TaskStatus) -> None:
        task.status = status
        if self._store is not None:
            self._store.update_task_status(task.id, status)

    def task_done(self) -> None:
        self._queue.task_done()

    async def join(self) -> None:
        await self._queue.join()

    def empty(self) -> bool:
        return self._queue.empty()

    def qsize(self) -> int:
        return self._queue.qsize()
