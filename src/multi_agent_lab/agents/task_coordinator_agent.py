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
        EventType.TEST_EXECUTION_FAILED,
    )

    def __init__(
        self,
        name: str,
        bus,
        graph_store: TaskGraphStore,
        event_logger=None,
        max_retries: int = 2,
        max_fix_attempts: int = 2,
    ) -> None:
        super().__init__(name, bus, event_logger)
        self.graph_store = graph_store
        self.max_retries = max_retries
        self.max_fix_attempts = max_fix_attempts
        self._fix_attempts: dict[str, int] = {}

    async def handle_message(self, message: Message) -> None:
        """Apply task graph transitions caused by worker events."""
        graph = self.graph_store.get(message.correlation_id or "")
        task_id = str(message.content["task_id"])

        if message.type == EventType.TEST_EXECUTION_FAILED:
            await self._handle_execution_failed(message, graph, task_id)
            return

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

    async def _handle_execution_failed(
        self,
        message: Message,
        graph,
        task_id: str,
    ) -> None:
        """Create a coding fix task for failed controlled execution."""
        key = f"{message.correlation_id}:{task_id}"
        attempts = self._fix_attempts.get(key, 0)
        if attempts >= self.max_fix_attempts:
            blocked_task = graph.block_task(task_id, "Max fix attempts exceeded.")
            self.graph_store.persist(graph)
            await self.publish(
                EventType.TASK_BLOCKED,
                {
                    "task_id": task_id,
                    "status": blocked_task.status.value,
                    "reason": "Max fix attempts exceeded.",
                },
                source=message,
            )
            await self.publish(
                EventType.WORKFLOW_HALTED,
                {
                    "reason": "max_fix_attempts_exceeded",
                    "task_id": task_id,
                    "attempts": attempts,
                    "execution_failure": message.content,
                },
                source=message,
            )
            await self._publish_graph_updated(message)
            return

        self._fix_attempts[key] = attempts + 1
        fix_task = graph.add_task(
            TaskNode(
                title=f"Corregir fallo de ejecucion #{attempts + 1}",
                description="Modificar archivos del workspace segun stdout/stderr.",
                required_capability="coding",
                payload={
                    "type": "fix",
                    "execution_task_id": task_id,
                    "attempt": attempts + 1,
                    "max_fix_attempts": self.max_fix_attempts,
                    "failure": message.content,
                    "path": self._primary_focus_file(message),
                    "suggested_focus_files": message.content.get("suggested_focus_files", []),
                    "command_id": message.content.get("command_id", "pytest"),
                    "args": message.content.get("args", []),
                },
                dependencies={task_id},
                priority=7,
                status=TaskNodeStatus.READY,
            )
        )
        self.graph_store.persist(graph)
        payload = self._task_ready_payload(fix_task)
        await self.publish(EventType.FIX_REQUESTED, payload, source=message)
        await self._publish_graph_updated(message)

    def _primary_focus_file(self, message: Message) -> str:
        focus_files = message.content.get("suggested_focus_files", [])
        if isinstance(focus_files, list) and focus_files:
            return str(focus_files[0])
        failed_files = message.content.get("failed_files", [])
        if isinstance(failed_files, list) and failed_files:
            return str(failed_files[0])
        return "app.py"

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
