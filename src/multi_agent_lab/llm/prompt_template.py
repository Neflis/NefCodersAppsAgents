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

    def render(self) -> str:
        """Render a deterministic prompt for JSON output."""
        payload = {
            "identity": self.identity,
            "capabilities": self.capabilities,
            "constraints": self.constraints,
            "input_context": self.input_context,
            "expected_json_output": self.expected_json_output,
        }
        return (
            "You are a local autonomous agent. Return only valid JSON.\n"
            f"{json.dumps(payload, ensure_ascii=True, indent=2)}"
        )
