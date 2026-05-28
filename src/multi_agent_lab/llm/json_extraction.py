"""Utilities for robust JSON extraction from LLM text."""

from __future__ import annotations

import json
import re
from typing import Any

from multi_agent_lab.llm.ollama_client_types import InvalidJSONError


class JsonExtractionService:
    """Extract, repair, and validate small JSON objects."""

    def extract_first_json(self, text: str) -> str:
        """Return the first balanced JSON object found in text."""
        start = text.find("{")
        if start < 0:
            raise InvalidJSONError("No JSON object found in LLM response.")

        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            if char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
        raise InvalidJSONError("Unbalanced JSON object in LLM response.")

    def repair_common_json_errors(self, text: str) -> str:
        """Repair common small JSON mistakes without guessing structure."""
        repaired = text.strip()
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        repaired = repaired.replace("True", "true").replace("False", "false")
        repaired = repaired.replace("None", "null")
        repaired = re.sub(r"'([^']+)'\s*:", r'"\1":', repaired)
        repaired = re.sub(r":\s*'([^']*)'", r': "\1"', repaired)
        return repaired

    def strip_markdown_fences(self, text: str) -> str:
        """Remove Markdown code fences around a JSON response."""
        stripped = text.strip()
        if not stripped.startswith("```"):
            return stripped
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
        return stripped.strip()

    def validate_against_schema(
        self,
        data: dict[str, Any],
        required_fields: list[str] | tuple[str, ...],
    ) -> dict[str, Any]:
        """Validate that required top-level fields exist."""
        missing = [field for field in required_fields if field not in data]
        if missing:
            raise InvalidJSONError(f"LLM JSON missing fields: {', '.join(missing)}")
        return data

    def parse(
        self,
        text: str,
        required_fields: list[str] | tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Parse JSON using direct, fenced, embedded, and repaired strategies."""
        candidates = [
            text,
            self.strip_markdown_fences(text),
        ]
        try:
            candidates.append(self.extract_first_json(text))
        except InvalidJSONError:
            pass

        last_error: Exception | None = None
        for candidate in candidates:
            for variant in (candidate, self.repair_common_json_errors(candidate)):
                try:
                    parsed = json.loads(variant)
                    if not isinstance(parsed, dict):
                        raise InvalidJSONError("LLM JSON response must be an object.")
                    return self.validate_against_schema(parsed, required_fields)
                except (json.JSONDecodeError, InvalidJSONError) as error:
                    last_error = error
        raise InvalidJSONError(f"Invalid JSON from LLM: {last_error}") from last_error
