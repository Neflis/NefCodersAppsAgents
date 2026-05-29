"""Failure analysis for controlled pytest execution."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from multi_agent_lab.core.file_path_normalizer import FilePathNormalizer
from multi_agent_lab.core.workspace_manager import WorkspaceSecurityError


class FixStrategy(StrEnum):
    """Supported fix strategies."""

    PATCH_EXISTING_FILE = "patch_existing_file"
    ADD_MISSING_IMPORT = "add_missing_import"
    ADD_MISSING_DEPENDENCY = "add_missing_dependency"
    FIX_ROUTE = "fix_route"
    FIX_TEST = "fix_test"
    REWRITE_FUNCTION = "rewrite_function"


@dataclass(slots=True)
class FailureContext:
    """Structured context extracted from a test or command failure."""

    failure_type: str
    failing_test: str = ""
    failing_line: int | None = None
    traceback: str = ""
    suspected_files: list[str] = field(default_factory=list)
    suspected_symbols: list[str] = field(default_factory=list)
    retry_number: int = 0

    def to_dict(self) -> dict[str, object]:
        """Serialize failure context."""
        return {
            "failure_type": self.failure_type,
            "failing_test": self.failing_test,
            "failing_line": self.failing_line,
            "traceback": self.traceback,
            "suspected_files": self.suspected_files,
            "suspected_symbols": self.suspected_symbols,
            "retry_number": self.retry_number,
        }


class FailureAnalysisService:
    """Parse pytest output into actionable failure context."""

    known_failures = (
        "ModuleNotFoundError",
        "ImportError",
        "SyntaxError",
        "AssertionError",
        "AttributeError",
        "NameError",
    )

    def __init__(self, path_normalizer: FilePathNormalizer | None = None) -> None:
        self.path_normalizer = path_normalizer or FilePathNormalizer()

    def parse_pytest_output(
        self,
        stdout: str,
        stderr: str,
        retry_number: int = 0,
    ) -> FailureContext:
        """Parse pytest stdout/stderr into a FailureContext."""
        combined = f"{stdout}\n{stderr}".strip()
        traceback = "\n\n".join(self.extract_tracebacks(combined))
        failure_type = self.detect_failure_type(combined)
        failing_test = self._extract_failing_test(combined)
        return FailureContext(
            failure_type=failure_type,
            failing_test=failing_test,
            failing_line=self._extract_failing_line(combined),
            traceback=traceback or combined[-4000:],
            suspected_files=self.infer_related_files(combined),
            suspected_symbols=self._extract_symbols(combined),
            retry_number=retry_number,
        )

    def extract_tracebacks(self, text: str) -> list[str]:
        """Extract traceback-like sections from pytest output."""
        sections: list[str] = []
        traceback_pattern = r"(Traceback \(most recent call last\):.*?)(?=\n\n|\Z)"
        for match in re.finditer(traceback_pattern, text, re.S):
            sections.append(match.group(1).strip())
        if not sections and "E   " in text:
            sections.append("\n".join(line for line in text.splitlines() if line.strip()))
        return sections

    def detect_failure_type(self, text: str) -> str:
        """Detect common failure type names."""
        lowered = text.lower()
        if "no module named" in lowered:
            return "ModuleNotFoundError"
        if "flask" in lowered and ("route" in lowered or "404" in lowered):
            return "FlaskRouteError"
        if "test_client" in lowered:
            return "MissingAppTestClient"
        for failure in self.known_failures:
            if failure in text:
                return failure
        if "failed" in lowered:
            return "AssertionError"
        return "UnknownFailure"

    def infer_related_files(self, text: str) -> list[str]:
        """Infer workspace files mentioned by failure output."""
        files = []
        candidates = self._path_candidates(text)
        for path in candidates:
            try:
                normalized = self.path_normalizer.normalize_workspace_path(path)
            except WorkspaceSecurityError:
                continue
            if normalized not in files:
                files.append(normalized)
        if "No module named" in text and "requirements.txt" not in files:
            files.insert(0, "requirements.txt")
        return files

    def summarize_failure(self, context: FailureContext) -> str:
        """Return a compact failure summary."""
        parts = [context.failure_type]
        if context.failing_test:
            parts.append(f"test={context.failing_test}")
        if context.suspected_files:
            parts.append(f"files={', '.join(context.suspected_files)}")
        return " | ".join(parts)

    def _extract_failing_test(self, text: str) -> str:
        match = re.search(r"FAILED\s+([^\s]+)", text)
        return match.group(1) if match else ""

    def _extract_failing_line(self, text: str) -> int | None:
        match = re.search(r":(\d+):", text)
        return int(match.group(1)) if match else None

    def _extract_symbols(self, text: str) -> list[str]:
        symbols = []
        for pattern in (r"name '([^']+)' is not defined", r"No module named '([^']+)'"):
            for match in re.finditer(pattern, text):
                if match.group(1) not in symbols:
                    symbols.append(match.group(1))
        return symbols

    def _path_candidates(self, text: str) -> list[str]:
        """Return raw path-like candidates from pytest output."""
        candidates: list[str] = []
        patterns = (
            r'File "([^"]+)", line \d+',
            r"FAILED\s+([^\s:]+\.py)(?:::|:)",
            r"((?:[A-Za-z]:)?[^\s'\"<>|]+(?:\.py|README\.md|requirements\.txt))",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                candidate = match.group(1)
                if candidate not in candidates:
                    candidates.append(candidate)
        return candidates
