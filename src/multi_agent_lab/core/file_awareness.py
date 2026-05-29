"""Workspace file awareness helpers."""

from __future__ import annotations

import logging
from pathlib import Path

from multi_agent_lab.core.file_path_normalizer import FilePathNormalizer
from multi_agent_lab.core.workspace_manager import WorkspaceSecurityError
from multi_agent_lab.tools.file_tool import FileTool, FileTooLargeError, FileToolError

logger = logging.getLogger(__name__)


class FileAwarenessService:
    """Summarizes safe workspace files for agent context and review."""

    def __init__(self, file_tool: FileTool, max_read_bytes: int | None = None) -> None:
        self.file_tool = file_tool
        self.max_read_bytes = max_read_bytes or file_tool.max_file_size_bytes
        self.path_normalizer = FilePathNormalizer(file_tool.workspace.root)
        self.invalid_paths_detected = 0
        self.invalid_paths_ignored = 0

    def list_files(self, path: str = ".") -> list[str]:
        """List safe workspace files."""
        return self.file_tool.list_files(path)

    def read_relevant_files(self, paths: list[str]) -> dict[str, str]:
        """Read relevant files without exceeding the configured safe limit."""
        contents: dict[str, str] = {}
        for path in paths:
            try:
                normalized = self.path_normalizer.normalize_workspace_path(path)
                safe_path = self.file_tool.workspace.resolve_safe_path(normalized)
            except WorkspaceSecurityError as error:
                self.invalid_paths_detected += 1
                self.invalid_paths_ignored += 1
                logger.warning("Ignoring invalid workspace context path %r: %s", path, error)
                continue
            if not safe_path.exists() or not safe_path.is_file():
                continue
            if safe_path.stat().st_size > self.max_read_bytes:
                raise FileTooLargeError(f"File is too large to summarize: {path}")
            contents[normalized] = self.file_tool.read_file(normalized)
        return contents

    def summarize_workspace(self) -> dict[str, object]:
        """Return a small workspace structure summary."""
        files = self.list_files(".")
        return {
            "root": str(self.file_tool.workspace.root),
            "files": files,
            "extensions": sorted({Path(path).suffix for path in files}),
        }

    def safe_read_optional(self, path: str) -> str:
        """Read one file, returning an empty string when it is absent."""
        try:
            return self.read_relevant_files([path]).get(path, "")
        except FileToolError:
            return ""
