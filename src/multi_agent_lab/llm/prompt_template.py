"""Prompt templates for agent reasoning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """Reusable prompt shape for structured agent decisions."""

    identity: str
    capabilities: list[str]
    constraints: list[str]
    input_context: dict[str, Any]
    expected_json_output: dict[str, Any]
    example_json_output: dict[str, Any] | None = None

    def render(self) -> str:
        """Render a compact deterministic prompt for JSON output."""
        payload = {
            "agent": self.identity,
            "capabilities": self.capabilities,
            "constraints": self.constraints,
            "context": self.input_context,
            "required_json": self.expected_json_output,
            "example": self.example_json_output or self.expected_json_output,
        }
        return (
            "Respond ONLY with valid JSON. No markdown. No prose. "
            "No code fences. No explanations outside JSON.\n"
            f"{json.dumps(payload, ensure_ascii=True, separators=(',', ':'))}"
        )
