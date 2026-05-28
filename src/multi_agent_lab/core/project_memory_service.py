"""Project memory service updated from workflow events."""

from __future__ import annotations

import json

from multi_agent_lab.core.message import EventType, Message
from multi_agent_lab.core.project_memory import ProjectMemory
from multi_agent_lab.core.sqlite_store import SQLiteStore


class ProjectMemoryService:
    """Maintains compressed semantic memory per correlation id."""

    relevant_events = {
        EventType.GOAL_DECOMPOSED,
        EventType.CODE_PROPOSED,
        EventType.FILE_WRITTEN,
        EventType.PROJECT_REVIEW_APPROVED,
        EventType.PROJECT_REVIEW_REJECTED,
        EventType.TEST_PASSED,
        EventType.TEST_FAILED,
    }

    def __init__(self, store: SQLiteStore | None = None) -> None:
        self.store = store
        self._memories: dict[str, ProjectMemory] = {}

    def get_memory(self, correlation_id: str) -> ProjectMemory:
        """Return existing memory or create an empty memory."""
        if correlation_id in self._memories:
            return self._memories[correlation_id]
        if self.store is not None:
            data = self.store.load_project_memory(correlation_id)
            if data is not None:
                memory = ProjectMemory.from_dict(data)
                self._memories[correlation_id] = memory
                return memory
        memory = ProjectMemory(correlation_id=correlation_id)
        self._memories[correlation_id] = memory
        self._persist(memory)
        return memory

    def update_from_event(self, message: Message) -> ProjectMemory | None:
        """Update memory from one relevant event."""
        try:
            event_type = EventType(str(message.type))
        except ValueError:
            return None
        if event_type not in self.relevant_events:
            return None
        correlation_id = message.correlation_id or message.id
        memory = self.get_memory(correlation_id)
        content = message.content if isinstance(message.content, dict) else {}

        if event_type == EventType.GOAL_DECOMPOSED:
            goal_text = content.get("goal")
            if isinstance(goal_text, str):
                memory.goal_summary = goal_text
            graph = content.get("graph", {})
            goal = graph.get("goal", {}) if isinstance(graph, dict) else {}
            title = goal.get("title") if isinstance(goal, dict) else None
            if isinstance(title, str):
                memory.goal_summary = title
            self._detect_framework(memory, json.dumps(content, ensure_ascii=True))

        if event_type == EventType.CODE_PROPOSED:
            path = content.get("path")
            code = content.get("content")
            if isinstance(path, str):
                memory.add_unique("architecture_decisions", f"Generated {path}")
            if isinstance(code, str):
                self._detect_framework(memory, code)
                self._detect_conventions(memory, code)

        if event_type == EventType.FILE_WRITTEN:
            path = content.get("path")
            if isinstance(path, str):
                self.add_file(correlation_id, path)
                memory = self.get_memory(correlation_id)

        if event_type == EventType.PROJECT_REVIEW_APPROVED:
            self.add_decision(correlation_id, "Project review approved.")
            memory = self.get_memory(correlation_id)
            memory.add_unique("reviewer_feedback", "Project review approved.")

        if event_type == EventType.PROJECT_REVIEW_REJECTED:
            feedback = content.get("feedback", [])
            for item in feedback if isinstance(feedback, list) else [feedback]:
                if isinstance(item, str):
                    self.add_error(correlation_id, item)
                    memory = self.get_memory(correlation_id)
                    memory.add_unique("reviewer_feedback", item)

        if event_type == EventType.TEST_PASSED:
            paths = content.get("paths") or content.get("path") or content.get("result")
            memory.add_unique("completed_tasks_summary", f"Mock validation passed: {paths}")

        if event_type == EventType.TEST_FAILED:
            error = content.get("error") or content.get("missing") or "Mock validation failed."
            self.add_error(correlation_id, str(error))
            memory = self.get_memory(correlation_id)

        memory.touch()
        self._persist(memory)
        return memory

    def summarize_for_context(self, correlation_id: str, max_chars: int = 1200) -> str:
        """Return a compact textual memory summary bounded by max_chars."""
        memory = self.get_memory(correlation_id)
        lines = [
            f"Goal: {memory.goal_summary or '(unknown)'}",
            f"Framework: {memory.detected_framework or '(unknown)'}",
            f"Files: {', '.join(memory.files_created) or '(none)'}",
            f"Decisions: {'; '.join(memory.architecture_decisions[-5:]) or '(none)'}",
            f"Conventions: {'; '.join(memory.coding_conventions[-5:]) or '(none)'}",
            f"Reviewer feedback: {'; '.join(memory.reviewer_feedback[-5:]) or '(none)'}",
            f"Known errors: {'; '.join(memory.known_errors[-5:]) or '(none)'}",
            f"Completed: {'; '.join(memory.completed_tasks_summary[-5:]) or '(none)'}",
        ]
        summary = "\n".join(lines)
        if len(summary) <= max_chars:
            return summary
        return summary[: max(0, max_chars - 3)].rstrip() + "..."

    def add_decision(self, correlation_id: str, decision: str) -> None:
        """Add one architecture decision."""
        memory = self.get_memory(correlation_id)
        memory.add_unique("architecture_decisions", decision)
        self._persist(memory)

    def add_error(self, correlation_id: str, error: str) -> None:
        """Add one known project error."""
        memory = self.get_memory(correlation_id)
        memory.add_unique("known_errors", error)
        self._persist(memory)

    def add_file(self, correlation_id: str, path: str) -> None:
        """Add one created file."""
        memory = self.get_memory(correlation_id)
        memory.add_unique("files_created", path)
        self._persist(memory)

    def _persist(self, memory: ProjectMemory) -> None:
        if self.store is not None:
            self.store.save_project_memory(memory.to_dict())

    def _detect_framework(self, memory: ProjectMemory, text: str) -> None:
        lowered = text.lower()
        if "flask" in lowered:
            memory.detected_framework = "Flask"
            memory.touch()

    def _detect_conventions(self, memory: ProjectMemory, code: str) -> None:
        if "from flask import" in code:
            memory.add_unique("coding_conventions", "Use Flask route decorators.")
        if "jsonify" in code:
            memory.add_unique("coding_conventions", "Return JSON responses with jsonify.")
