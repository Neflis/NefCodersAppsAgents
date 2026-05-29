"""Normalize traceback paths into safe workspace-relative paths."""

from __future__ import annotations

from pathlib import Path

from multi_agent_lab.core.workspace_manager import WorkspaceSecurityError


class FilePathNormalizer:
    """Converts candidate paths into safe workspace-relative paths."""

    def __init__(self, workspace_root: str | Path | None = None) -> None:
        self.workspace_root = Path(workspace_root).resolve() if workspace_root else None

    def normalize_workspace_path(self, path: str | Path) -> str:
        """Return a normalized workspace-relative path or raise if unsafe."""
        stripped = self.strip_absolute_prefix(path)
        return self.ensure_relative_workspace_path(stripped)

    def strip_absolute_prefix(self, path: str | Path) -> str:
        """Strip a trusted workspace absolute prefix from a path."""
        raw = self._clean(path)
        candidate = Path(raw)
        if not candidate.is_absolute():
            return raw.replace("\\", "/")

        if self.workspace_root is not None:
            resolved = candidate.resolve(strict=False)
            if resolved == self.workspace_root or self.workspace_root in resolved.parents:
                return resolved.relative_to(self.workspace_root).as_posix()

        parts = candidate.parts
        normalized_parts = [part.strip("\\/") for part in parts]
        for index, part in enumerate(normalized_parts):
            if part == "workspace" and index + 1 < len(normalized_parts):
                return "/".join(normalized_parts[index + 1 :])

        self.reject_external_path(raw)
        return raw

    def reject_external_path(self, path: str | Path) -> None:
        """Raise for a path that points outside the workspace."""
        raise WorkspaceSecurityError(f"Path is outside the workspace: {path}")

    def ensure_relative_workspace_path(self, path: str | Path) -> str:
        """Ensure the path is relative, traversal-free, and safe for workspace use."""
        cleaned = self._clean(path).replace("\\", "/")
        candidate = Path(cleaned)
        if not cleaned or cleaned in {".", "/"}:
            raise WorkspaceSecurityError("Empty workspace path is not allowed.")
        if "\x00" in cleaned:
            raise WorkspaceSecurityError("Path contains a null byte.")
        if candidate.is_absolute():
            raise WorkspaceSecurityError("Absolute paths are not allowed.")
        if any(part in {"", ".."} for part in candidate.parts):
            raise WorkspaceSecurityError("Path traversal is not allowed.")
        if cleaned.startswith("../") or "/../" in cleaned:
            raise WorkspaceSecurityError("Path traversal is not allowed.")
        return candidate.as_posix()

    def _clean(self, path: str | Path) -> str:
        text = str(path).strip().strip("\"'`")
        text = text.rstrip("),:]")
        if text.startswith("./") or text.startswith(".\\"):
            text = text[2:]
        return text
