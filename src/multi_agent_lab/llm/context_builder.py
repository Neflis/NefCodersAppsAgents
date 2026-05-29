"""Context builder for LLM agent prompts."""

from __future__ import annotations

import json
import logging
from typing import Any

from multi_agent_lab.core.file_awareness import FileAwarenessService
from multi_agent_lab.core.message import Message
from multi_agent_lab.core.project_memory_service import ProjectMemoryService
from multi_agent_lab.core.task_graph_store import TaskGraphStore
from multi_agent_lab.core.workspace_manager import WorkspaceSecurityError
from multi_agent_lab.tools.file_tool import FileTool, FileToolError

logger = logging.getLogger(__name__)


class AgentContextBuilder:
    """Build compact context from graph state, events, and workspace files."""

    def __init__(
        self,
        graph_store: TaskGraphStore | None = None,
        file_tool: FileTool | None = None,
        file_awareness: FileAwarenessService | None = None,
        project_memory: ProjectMemoryService | None = None,
        recent_events_limit: int = 8,
        max_context_chars: int = 6000,
    ) -> None:
        self.graph_store = graph_store
        self.file_tool = file_tool
        self.file_awareness = file_awareness or (
            FileAwarenessService(file_tool) if file_tool is not None else None
        )
        self.project_memory = project_memory
        self.recent_events_limit = recent_events_limit
        self.max_context_chars = max_context_chars
        self._events_by_correlation: dict[str, list[dict[str, Any]]] = {}

    def record_event(self, message: Message) -> None:
        """Record one recent event for future context."""
        correlation_id = message.correlation_id or message.id
        events = self._events_by_correlation.setdefault(correlation_id, [])
        events.append(
            {
                "id": message.id,
                "type": str(message.type),
                "sender": message.sender,
                "content": message.content,
            }
        )
        del events[: -self.recent_events_limit]

    def build(
        self,
        message: Message,
        current_task: dict[str, Any] | None = None,
        workspace_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build prompt context for an agent."""
        correlation_id = message.correlation_id or message.id
        graph_summary: dict[str, Any] | None = None
        if self.graph_store is not None and correlation_id in self.graph_store:
            graph = self.graph_store.get(correlation_id)
            graph_summary = {
                "goal": graph.goal.to_dict(),
                "tasks": [
                    {
                        "id": node.id,
                        "title": node.title,
                        "status": node.status.value,
                        "required_capability": node.required_capability,
                        "dependencies": sorted(node.dependencies),
                        "retries": node.retries,
                    }
                    for node in graph.nodes.values()
                ],
            }

        files: dict[str, str] = {}
        workspace_tree: dict[str, object] = {"files": []}
        if self.file_awareness is not None:
            try:
                workspace_tree = self.file_awareness.summarize_workspace()
                files = self.file_awareness.read_relevant_files(workspace_paths or [])
            except (FileToolError, WorkspaceSecurityError) as error:
                logger.warning("Ignoring invalid workspace context paths: %s", error)
                files = {}

        memory_summary = ""
        if self.project_memory is not None:
            memory_summary = self.project_memory.summarize_for_context(
                correlation_id,
                max_chars=max(500, self.max_context_chars // 4),
            )

        context = {
            "global_goal": message.content.get("goal"),
            "current_event": {
                "id": message.id,
                "type": str(message.type),
                "content": message.content,
            },
            "current_task": current_task,
            "task_graph": graph_summary,
            "recent_events": self._events_by_correlation.get(correlation_id, []),
            "project_memory": memory_summary,
            "workspace_tree": workspace_tree,
            "workspace_files": files,
            "previous_decisions": self._recent_decision_events(correlation_id),
            "correlation_id": correlation_id,
        }
        return self._fit_context(context)

    def build_json(
        self,
        message: Message,
        current_task: dict[str, Any] | None = None,
        workspace_paths: list[str] | None = None,
    ) -> str:
        """Build context as JSON text."""
        return json.dumps(
            self.build(message, current_task, workspace_paths),
            ensure_ascii=True,
            indent=2,
        )

    def _recent_decision_events(self, correlation_id: str) -> list[dict[str, Any]]:
        """Return recent high-signal events for the same workflow."""
        decision_types = {
            "CODE_PROPOSED",
            "REVIEW_APPROVED",
            "REVIEW_REJECTED",
            "PROJECT_REVIEW_APPROVED",
            "PROJECT_REVIEW_REJECTED",
            "FILE_WRITTEN",
            "TEST_PASSED",
            "TEST_FAILED",
        }
        events = self._events_by_correlation.get(correlation_id, [])
        return [event for event in events if event["type"] in decision_types]

    def _fit_context(self, context: dict[str, Any]) -> dict[str, Any]:
        """Bound context size before sending it to an LLM."""
        rendered = json.dumps(context, ensure_ascii=True)
        if len(rendered) <= self.max_context_chars:
            return context

        compact = dict(context)
        compact["recent_events"] = compact.get("recent_events", [])[-3:]
        compact["previous_decisions"] = compact.get("previous_decisions", [])[-3:]
        files = compact.get("workspace_files", {})
        if isinstance(files, dict):
            compact["workspace_files"] = {
                path: self._truncate_text(str(content), 500) for path, content in files.items()
            }

        rendered = json.dumps(compact, ensure_ascii=True)
        if len(rendered) <= self.max_context_chars:
            return compact

        compact["task_graph"] = self._compact_task_graph(compact.get("task_graph"))
        rendered = json.dumps(compact, ensure_ascii=True)
        if len(rendered) <= self.max_context_chars:
            return compact

        compact["workspace_files"] = {}
        compact["context_truncated"] = True
        rendered = json.dumps(compact, ensure_ascii=True)
        if len(rendered) <= self.max_context_chars:
            return compact

        minimal = {
            "global_goal": compact.get("global_goal"),
            "current_task": compact.get("current_task"),
            "project_memory": self._truncate_text(
                str(compact.get("project_memory", "")),
                max(50, self.max_context_chars // 2),
            ),
            "workspace_tree": compact.get("workspace_tree"),
            "correlation_id": compact.get("correlation_id"),
            "context_truncated": True,
        }
        rendered = json.dumps(minimal, ensure_ascii=True)
        if len(rendered) <= self.max_context_chars:
            return minimal
        minimal["workspace_tree"] = {"files": []}
        minimal["project_memory"] = ""
        return minimal

    def _compact_task_graph(self, graph_summary: object) -> object:
        """Keep only compact task graph fields when context is too large."""
        if not isinstance(graph_summary, dict):
            return graph_summary
        tasks = graph_summary.get("tasks", [])
        if not isinstance(tasks, list):
            return graph_summary
        return {
            "goal": graph_summary.get("goal"),
            "tasks": [
                {
                    "title": task.get("title"),
                    "status": task.get("status"),
                    "required_capability": task.get("required_capability"),
                }
                for task in tasks
                if isinstance(task, dict)
            ],
        }

    def _truncate_text(self, text: str, max_chars: int) -> str:
        """Truncate a text field safely."""
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."
