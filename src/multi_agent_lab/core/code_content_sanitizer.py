"""Sanitize generated code content before workspace writes."""

from __future__ import annotations

import re
from pathlib import Path


class CodeContentSanitizer:
    """Remove markdown fences from code-like files while preserving README markdown."""

    code_extensions = {".py", ".java", ".json", ".xml", ".txt", ".ts", ".html"}

    def __init__(self) -> None:
        self.sanitized_files_count = 0

    def detect_markdown_fences(self, content: str) -> bool:
        """Return whether content contains markdown code fences."""
        return "```" in content

    def strip_markdown_code_fences(self, content: str) -> str:
        """Remove opening and closing markdown code fence lines."""
        lines = content.splitlines()
        stripped_lines = [
            line
            for line in lines
            if not re.match(
                r"^\s*```(?:python|py|java|xml|json|txt|typescript|ts|html)?\s*$",
                line,
                re.I,
            )
        ]
        stripped = "\n".join(stripped_lines)
        if content.endswith("\n"):
            stripped += "\n"
        return stripped

    def normalize_code_content(self, path: str, content: str) -> str:
        """Normalize content for a target workspace path."""
        if not self._should_sanitize(path) or not self.detect_markdown_fences(content):
            return content
        sanitized = self.strip_markdown_code_fences(content)
        if sanitized != content:
            self.sanitized_files_count += 1
        return sanitized

    def validate_no_markdown_fences(self, path: str, content: str) -> None:
        """Raise when code content still contains markdown fences."""
        if self._should_sanitize(path) and self.detect_markdown_fences(content):
            raise ValueError("Python files must not include markdown code fences.")

    def _should_sanitize(self, path: str) -> bool:
        suffix = Path(path).suffix.lower()
        return suffix in self.code_extensions and Path(path).name != "README.md"
