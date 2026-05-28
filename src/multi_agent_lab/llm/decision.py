"""LLM decision model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class LLMDecision:
    """Structured decision returned by an LLM-capable agent."""

    action: str
    reasoning_summary: str
    confidence: float
    events_to_publish: list[dict[str, Any]] = field(default_factory=list)
    task_updates: list[dict[str, Any]] = field(default_factory=list)
    content: Any = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LLMDecision:
        """Build a decision from parsed JSON."""
        return cls(
            action=str(data.get("action", "noop")),
            reasoning_summary=str(data.get("reasoning_summary", "")),
            confidence=float(data.get("confidence", 0.0)),
            events_to_publish=list(data.get("events_to_publish", [])),
            task_updates=list(data.get("task_updates", [])),
            content=data.get("content"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the decision."""
        return {
            "action": self.action,
            "reasoning_summary": self.reasoning_summary,
            "confidence": self.confidence,
            "events_to_publish": self.events_to_publish,
            "task_updates": self.task_updates,
            "content": self.content,
        }
