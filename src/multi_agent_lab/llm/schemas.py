"""Required fields for structured agent LLM outputs."""

from __future__ import annotations

PLANNER_DECISION_FIELDS = ("tasks", "reasoning_summary")
CODER_DECISION_FIELDS = ("content", "reasoning_summary")
REVIEWER_DECISION_FIELDS = ("approved", "feedback", "reasoning_summary")
