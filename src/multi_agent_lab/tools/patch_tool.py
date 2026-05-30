"""Safe search/replace patches restricted to the workspace."""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path

from multi_agent_lab.core.workspace_manager import WorkspaceManager, WorkspaceSecurityError
from multi_agent_lab.tools.file_tool import MAX_FILE_SIZE_BYTES, UnsupportedFileTypeError


class PatchToolError(ValueError):
    """Raised when a patch is unsafe or cannot be applied."""


class PatchValidationError(PatchToolError):
    """Raised when a search/replace patch is invalid."""


@dataclass(slots=True)
class PatchPreview:
    """Preview of a safe patch before it is applied."""

    path: str
    original: str
    patched: str
    diff_summary: str
    diff: str


@dataclass(slots=True)
class PatchApplyResult:
    """Result of an applied safe patch."""

    path: str
    diff_summary: str
    diff: str


class PatchTool:
    """Apply exact search/replace patches to existing text files."""

    text_extensions = {
        ".py",
        ".md",
        ".txt",
        ".json",
        ".yaml",
        ".yml",
        ".xml",
        ".java",
        ".ts",
        ".html",
    }

    def __init__(
        self,
        workspace: WorkspaceManager,
        max_file_size_bytes: int = MAX_FILE_SIZE_BYTES,
    ) -> None:
        self.workspace = workspace
        self.max_file_size_bytes = max_file_size_bytes
        self._backups: dict[str, str] = {}

    def validate_patch(self, path: str | Path, search: str, replace: str) -> None:
        """Validate that a search/replace patch can be applied safely."""
        safe_path = self._resolve_existing_text_file(path)
        if not search:
            raise PatchValidationError("Search text cannot be empty.")
        if len(replace.encode("utf-8")) > self.max_file_size_bytes:
            raise PatchValidationError("Replacement text exceeds the maximum file size.")

        content = self._read_text(safe_path)
        count = content.count(search)
        if count == 0:
            raise PatchValidationError("Search text was not found exactly once.")
        if count > 1:
            raise PatchValidationError("Search text is ambiguous and appears more than once.")

    def preview_patch(self, path: str | Path, search: str, replace: str) -> PatchPreview:
        """Return the patched content and a compact diff without writing."""
        self.validate_patch(path, search, replace)
        safe_path = self._resolve_existing_text_file(path)
        original = self._read_text(safe_path)
        patched = original.replace(search, replace, 1)
        if len(patched.encode("utf-8")) > self.max_file_size_bytes:
            raise PatchValidationError("Patched file would exceed the maximum file size.")
        relative_path = self.workspace.relative_to_workspace(
            safe_path.relative_to(self.workspace.root)
        )
        diff = self._unified_diff(relative_path, original, patched)
        return PatchPreview(
            path=relative_path,
            original=original,
            patched=patched,
            diff_summary=self._diff_summary(relative_path, diff),
            diff=diff,
        )

    def apply_search_replace(
        self,
        path: str | Path,
        search: str,
        replace: str,
    ) -> PatchApplyResult:
        """Apply a validated search/replace patch to an existing file."""
        preview = self.preview_patch(path, search, replace)
        safe_path = self._resolve_existing_text_file(path)
        self._backups[preview.path] = preview.original
        safe_path.write_text(preview.patched, encoding="utf-8")
        return PatchApplyResult(
            path=preview.path,
            diff_summary=preview.diff_summary,
            diff=preview.diff,
        )

    def backup_for(self, path: str | Path) -> str | None:
        """Return the last internal backup recorded for a workspace path."""
        try:
            safe_path = self.workspace.resolve_safe_path(path)
            relative_path = self.workspace.relative_to_workspace(
                safe_path.relative_to(self.workspace.root)
            )
        except WorkspaceSecurityError:
            return None
        return self._backups.get(relative_path)

    def _resolve_existing_text_file(self, path: str | Path) -> Path:
        try:
            safe_path = self.workspace.resolve_safe_path(path)
        except WorkspaceSecurityError as error:
            raise PatchToolError(str(error)) from error
        if not safe_path.exists():
            raise PatchValidationError("PatchTool cannot create new files.")
        if not safe_path.is_file():
            raise PatchValidationError("Patch target must be a file.")
        if safe_path.suffix not in self.text_extensions:
            raise UnsupportedFileTypeError(f"Extension not allowed: {safe_path.suffix}")
        if safe_path.stat().st_size > self.max_file_size_bytes:
            raise PatchValidationError("Patch target exceeds the maximum file size.")
        self._ensure_text_file(safe_path)
        return safe_path

    def _ensure_text_file(self, path: Path) -> None:
        sample = path.read_bytes()[:2048]
        if b"\x00" in sample:
            raise PatchValidationError("Binary files cannot be patched.")
        try:
            sample.decode("utf-8")
        except UnicodeDecodeError as error:
            raise PatchValidationError("Binary files cannot be patched.") from error

    def _read_text(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise PatchValidationError("Binary files cannot be patched.") from error

    def _unified_diff(self, path: str, original: str, patched: str) -> str:
        return "".join(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                patched.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
            )
        )

    def _diff_summary(self, path: str, diff: str) -> str:
        added = 0
        removed = 0
        for line in diff.splitlines():
            if line.startswith("+++") or line.startswith("---"):
                continue
            if line.startswith("+"):
                added += 1
            if line.startswith("-"):
                removed += 1
        return f"{path}: +{added} -{removed}"
