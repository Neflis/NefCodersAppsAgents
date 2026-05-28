"""Task coordinator agent for dynamic task graph updates."""

from __future__ import annotations

import logging

from multi_agent_lab.agents.base_agent import BaseAgent
from multi_agent_lab.core.message import EventType, Message
from multi_agent_lab.core.task_graph import TaskNode, TaskNodeStatus
from multi_agent_lab.core.task_graph_store import TaskGraphStore

logger = logging.getLogger(__name__)


class TaskCoordinatorAgent(BaseAgent):
    """Agent that unlocks dependencies and manages retries."""

    subscribed_events = (
        EventType.TASK_CLAIMED,
        EventType.TASK_COMPLETED,
        EventType.TASK_FAILED,
    )

    def __init__(
        self,
        name: str,
        bus,
        graph_store: TaskGraphStore,
        event_logger=None,
        max_retries: int = 2,
    ) -> None:
        super().__init__(name, bus, event_logger)
        self.graph_store = graph_store
        self.max_retries = max_retries

    async def handle_message(self, message: Message) -> None:
        """Apply task graph transitions caused by worker events."""
        graph = self.graph_store.get(message.correlation_id or "")
        task_id = str(message.content["task_id"])

        if message.type == EventType.TASK_CLAIMED:
            graph.claim_task(task_id, str(message.content["owner"]))
            self.graph_store.persist(graph)
            await self._publish_graph_updated(message)
            return

        if message.type == EventType.TASK_COMPLETED:
            graph.complete_task(task_id, dict(message.content.get("result", {})))
            self.graph_store.persist(graph)
            await self._publish_graph_updated(message)
            for ready_task in graph.ready_tasks():
                self.graph_store.persist(graph)
                await self.publish(
                    EventType.TASK_READY,
                    self._task_ready_payload(ready_task),
                    source=message,
                )
            return

        if message.type == EventType.TASK_FAILED:
            failed_task = graph.fail_task(
                task_id, str(message.content.get("error", "unknown error"))
            )
            if failed_task.retries < min(failed_task.max_retries, self.max_retries):
                retried_task = graph.retry_task(task_id)
                self.graph_store.persist(graph)
                await self.publish(
                    EventType.TASK_RETRIED,
                    {"task_id": task_id, "retries": retried_task.retries},
                    source=message,
                )
                await self.publish(
                    EventType.TASK_READY,
                    self._task_ready_payload(retried_task),
                    source=message,
                )
            else:
                blocked_task = graph.block_task(task_id, "Max retries exceeded.")
                self.graph_store.persist(graph)
                await self.publish(
                    EventType.TASK_BLOCKED,
                    {"task_id": task_id, "status": blocked_task.status.value},
                    source=message,
                )
            await self._publish_graph_updated(message)

    async def _publish_graph_updated(self, source: Message) -> None:
        graph = self.graph_store.get(source.correlation_id or "")
        await self.publish(EventType.TASK_GRAPH_UPDATED, {"graph": graph.to_dict()}, source=source)

    def _task_ready_payload(self, task: TaskNode) -> dict[str, object]:
        """Build a TASK_READY payload."""
        return {
            "task_id": task.id,
            "title": task.title,
            "required_capability": task.required_capability,
            "payload": task.payload,
            "status": TaskNodeStatus.READY.value,
            "priority": task.priority,
        }
