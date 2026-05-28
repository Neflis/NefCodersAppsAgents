"""Semantic project memory for one workflow correlation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def utc_now_iso() -> str:
    """Return a compact UTC timestamp."""
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class ProjectMemory:
    """Compressed semantic memory for a project workflow."""

    correlation_id: str
    goal_summary: str = ""
    detected_framework: str = ""
    files_created: list[str] = field(default_factory=list)
    architecture_decisions: list[str] = field(default_factory=list)
    coding_conventions: list[str] = field(default_factory=list)
    known_errors: list[str] = field(default_factory=list)
    reviewer_feedback: list[str] = field(default_factory=list)
    completed_tasks_summary: list[str] = field(default_factory=list)
    updated_at: str = field(default_factory=utc_now_iso)

    def touch(self) -> None:
        """Refresh the update timestamp."""
        self.updated_at = utc_now_iso()

    def add_unique(self, field_name: str, value: str) -> None:
        """Append a non-empty value to a list field once."""
        if not value:
            return
        values = getattr(self, field_name)
        if value not in values:
            values.append(value)
            self.touch()

    def to_dict(self) -> dict[str, Any]:
        """Serialize memory to primitive values."""
        return {
            "correlation_id": self.correlation_id,
            "goal_summary": self.goal_summary,
            "detected_framework": self.detected_framework,
            "files_created": self.files_created,
            "architecture_decisions": self.architecture_decisions,
            "coding_conventions": self.coding_conventions,
            "known_errors": self.known_errors,
            "reviewer_feedback": self.reviewer_feedback,
            "completed_tasks_summary": self.completed_tasks_summary,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectMemory:
        """Deserialize memory from primitive values."""
        return cls(
            correlation_id=str(data["correlation_id"]),
            goal_summary=str(data.get("goal_summary", "")),
            detected_framework=str(data.get("detected_framework", "")),
            files_created=list(data.get("files_created", [])),
            architecture_decisions=list(data.get("architecture_decisions", [])),
            coding_conventions=list(data.get("coding_conventions", [])),
            known_errors=list(data.get("known_errors", [])),
            reviewer_feedback=list(data.get("reviewer_feedback", [])),
            completed_tasks_summary=list(data.get("completed_tasks_summary", [])),
            updated_at=str(data.get("updated_at", utc_now_iso())),
        )
