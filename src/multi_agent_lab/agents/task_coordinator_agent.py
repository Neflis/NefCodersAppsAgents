"""Task coordinator agent for dynamic task graph updates."""

from __future__ import annotations

import logging

from multi_agent_lab.agents.base_agent import BaseAgent
from multi_agent_lab.core.failure_analysis import FailureAnalysisService, FixStrategy
from multi_agent_lab.core.file_path_normalizer import FilePathNormalizer
from multi_agent_lab.core.message import EventType, Message
from multi_agent_lab.core.task_graph import TaskNode, TaskNodeStatus
from multi_agent_lab.core.task_graph_store import TaskGraphStore
from multi_agent_lab.core.workspace_manager import WorkspaceSecurityError

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
        path_normalizer: FilePathNormalizer | None = None,
    ) -> None:
        super().__init__(name, bus, event_logger)
        self.graph_store = graph_store
        self.max_retries = max_retries
        self.max_fix_attempts = max_fix_attempts
        self._fix_attempts: dict[str, int] = {}
        self.path_normalizer = path_normalizer or FilePathNormalizer()
        self.failure_analysis = FailureAnalysisService(self.path_normalizer)

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
        failure_context = self.failure_analysis.parse_pytest_output(
            str(message.content.get("stdout", "")),
            str(message.content.get("stderr", "")),
            retry_number=attempts + 1,
        )
        failure_context_data = self._safe_failure_context(failure_context.to_dict())
        suggested_focus_files = self._safe_workspace_files(
            message.content.get("suggested_focus_files", [])
        )
        fix_strategy = self._fix_strategy(failure_context_data)
        target_files = self._target_files(failure_context_data, suggested_focus_files)
        fix_metadata = self._fix_metadata(
            message,
            attempts + 1,
            failure_context_data,
            target_files,
            fix_strategy,
        )
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
                    "failure_context": failure_context_data,
                    "failure_summary": self.failure_analysis.summarize_failure(failure_context),
                    "path": self._primary_focus_file(message, failure_context_data, target_files),
                    "suggested_focus_files": target_files,
                    "target_files": target_files,
                    "fix_strategy": fix_strategy,
                    "command_id": message.content.get("command_id", "pytest"),
                    "args": message.content.get("args", []),
                    "metadata": fix_metadata,
                },
                dependencies={task_id},
                priority=7,
                status=TaskNodeStatus.READY,
            )
        )
        self.graph_store.persist(graph)
        payload = self._task_ready_payload(fix_task)
        await self.publish(EventType.FIX_REQUESTED, payload, source=message, metadata=fix_metadata)
        await self.publish(EventType.TASK_READY, payload, source=message, metadata=fix_metadata)
        await self._publish_graph_updated(message)

    def _fix_metadata(
        self,
        message: Message,
        attempt: int,
        failure_context: dict[str, object],
        target_files: list[str],
        fix_strategy: str,
    ) -> dict[str, object]:
        """Build metadata for a generated fix task."""
        return {
            "failed_command": message.content.get("command", ""),
            "stdout": message.content.get("stdout", ""),
            "stderr": message.content.get("stderr", ""),
            "suggested_focus_files": target_files,
            "target_files": target_files,
            "fix_strategy": fix_strategy,
            "fix_attempt": attempt,
            "failure_context": failure_context,
            "failure_analysis": failure_context,
        }

    def _primary_focus_file(
        self,
        message: Message,
        failure_context: dict[str, object] | None = None,
        suggested_focus_files: list[str] | None = None,
    ) -> str:
        focus_files = suggested_focus_files
        if focus_files:
            return str(focus_files[0])
        context = failure_context or message.content.get("failure_context", {})
        if isinstance(context, dict):
            suspected_files = context.get("suspected_files", [])
            if isinstance(suspected_files, list) and suspected_files:
                return str(suspected_files[0])
        failed_files = message.content.get("failed_files", [])
        if isinstance(failed_files, list) and failed_files:
            safe_failed_files = self._safe_workspace_files(failed_files)
            if safe_failed_files:
                return safe_failed_files[0]
        return "app.py"

    def _fix_strategy(self, failure_context: dict[str, object]) -> str:
        """Return the strategy implied by failure context."""
        failure_type = str(failure_context.get("failure_type", ""))
        if failure_type == "LocalModuleNotFoundError":
            return FixStrategy.FIX_LOCAL_MODULE_IMPORT.value
        if failure_type == "ModuleNotFoundError":
            return FixStrategy.ADD_MISSING_DEPENDENCY.value
        return ""

    def _target_files(
        self,
        failure_context: dict[str, object],
        suggested_focus_files: list[str],
    ) -> list[str]:
        """Choose target files from failure type without drifting to requirements."""
        if failure_context.get("failure_type") == "LocalModuleNotFoundError":
            suspected = failure_context.get("suspected_files", [])
            local_targets = [
                str(path) for path in suspected if isinstance(path, str) and path.endswith(".py")
            ]
            for fallback in ("tests/test_app.py", "app.py"):
                if fallback not in local_targets:
                    local_targets.append(fallback)
            return local_targets
        return suggested_focus_files or self._safe_workspace_files(
            failure_context.get("suspected_files", [])
        )

    def _safe_failure_context(self, failure_context: dict[str, object]) -> dict[str, object]:
        """Filter paths inside a failure context."""
        safe_context = dict(failure_context)
        safe_context["suspected_files"] = self._safe_workspace_files(
            safe_context.get("suspected_files", [])
        )
        return safe_context

    def _safe_workspace_files(self, paths: object) -> list[str]:
        """Return only normalized workspace-relative paths."""
        if not isinstance(paths, list):
            return []
        safe_paths: list[str] = []
        for path in paths:
            try:
                normalized = self.path_normalizer.normalize_workspace_path(str(path))
            except WorkspaceSecurityError as error:
                logger.warning("Ignoring invalid failure path %r: %s", path, error)
                continue
            if normalized not in safe_paths:
                safe_paths.append(normalized)
        return safe_paths

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
