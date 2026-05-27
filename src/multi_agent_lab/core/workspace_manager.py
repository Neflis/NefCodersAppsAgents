"""Safe workspace path management."""

from __future__ import annotations

from pathlib import Path


class WorkspaceSecurityError(ValueError):
    """Raised when a workspace path is unsafe."""


class WorkspaceManager:
    """Resolve and validate paths inside a controlled workspace."""

    dangerous_parts = {".git", ".hg", ".svn", "__pycache__", ".env"}

    def __init__(self, root: str | Path = "workspace") -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve_safe_path(self, path: str | Path) -> Path:
        """Return an absolute path guaranteed to be inside the workspace."""
        raw_path = Path(path)
        path_text = str(path)
        if "\x00" in path_text:
            raise WorkspaceSecurityError("Path contains a null byte.")
        if raw_path.is_absolute():
            raise WorkspaceSecurityError("Absolute paths are not allowed.")
        if any(part in {"..", ""} for part in raw_path.parts):
            raise WorkspaceSecurityError("Path traversal is not allowed.")
        if any(part in self.dangerous_parts for part in raw_path.parts):
            raise WorkspaceSecurityError("Dangerous path segment is not allowed.")

        resolved = (self.root / raw_path).resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise WorkspaceSecurityError("Resolved path escapes the workspace.")
        return resolved

    def relative_to_workspace(self, path: str | Path) -> str:
        """Return a safe workspace-relative path."""
        return self.resolve_safe_path(path).relative_to(self.root).as_posix()
