"""Context builder for LLM agent prompts."""

from __future__ import annotations

import json
from typing import Any

from multi_agent_lab.core.file_awareness import FileAwarenessService
from multi_agent_lab.core.message import Message
from multi_agent_lab.core.task_graph_store import TaskGraphStore
from multi_agent_lab.tools.file_tool import FileTool


class AgentContextBuilder:
    """Build compact context from graph state, events, and workspace files."""

    def __init__(
        self,
        graph_store: TaskGraphStore | None = None,
        file_tool: FileTool | None = None,
        file_awareness: FileAwarenessService | None = None,
        recent_events_limit: int = 8,
    ) -> None:
        self.graph_store = graph_store
        self.file_tool = file_tool
        self.file_awareness = file_awareness or (
            FileAwarenessService(file_tool) if file_tool is not None else None
        )
        self.recent_events_limit = recent_events_limit
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
            workspace_tree = self.file_awareness.summarize_workspace()
            files = self.file_awareness.read_relevant_files(workspace_paths or [])

        return {
            "global_goal": message.content.get("goal"),
            "current_event": {
                "id": message.id,
                "type": str(message.type),
                "content": message.content,
            },
            "current_task": current_task,
            "task_graph": graph_summary,
            "recent_events": self._events_by_correlation.get(correlation_id, []),
            "workspace_tree": workspace_tree,
            "workspace_files": files,
            "previous_decisions": self._events_by_correlation.get(correlation_id, []),
            "correlation_id": correlation_id,
        }

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
