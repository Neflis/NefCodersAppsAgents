"""Priority-aware async task queue."""

from __future__ import annotations

import asyncio
from itertools import count

from multi_agent_lab.core.task import Task


class TaskQueue:
    def __init__(self) -> None:
        self._queue: asyncio.PriorityQueue[tuple[int, int, Task]] = asyncio.PriorityQueue()
        self._sequence = count()

    async def put(self, task: Task) -> None:
        await self._queue.put((-task.priority, next(self._sequence), task))

    async def get(self) -> Task:
        _, _, task = await self._queue.get()
        return task

    def task_done(self) -> None:
        self._queue.task_done()

    async def join(self) -> None:
        await self._queue.join()

    def empty(self) -> bool:
        return self._queue.empty()

    def qsize(self) -> int:
        return self._queue.qsize()
