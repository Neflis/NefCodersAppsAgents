"""Safe file operations restricted to the project workspace."""

from __future__ import annotations

from pathlib import Path

from multi_agent_lab.core.workspace_manager import WorkspaceManager, WorkspaceSecurityError

MAX_FILE_SIZE_BYTES = 1024 * 1024


class FileToolError(ValueError):
    """Raised when a file operation is not allowed."""


class FileTooLargeError(FileToolError):
    """Raised when a file exceeds the configured size limit."""


class UnsupportedFileTypeError(FileToolError):
    """Raised when a file extension is not allowed."""


class FileTool:
    """Read and write allowed files inside a controlled workspace."""

    allowed_extensions = {".py", ".md", ".txt", ".json", ".yaml", ".yml"}

    def __init__(
        self,
        workspace: WorkspaceManager,
        max_file_size_bytes: int = MAX_FILE_SIZE_BYTES,
    ) -> None:
        self.workspace = workspace
        self.max_file_size_bytes = max_file_size_bytes

    def read_file(self, path: str | Path) -> str:
        """Read an allowed file from the workspace."""
        safe_path = self._resolve_allowed_file(path)
        if not safe_path.exists():
            raise FileToolError(f"File does not exist: {path}")
        if safe_path.stat().st_size > self.max_file_size_bytes:
            raise FileTooLargeError("File is too large to read.")
        return safe_path.read_text(encoding="utf-8")

    def write_file(self, path: str | Path, content: str) -> str:
        """Write content to an allowed file in the workspace."""
        safe_path = self._resolve_allowed_file(path)
        self._validate_content_size(content)
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        safe_path.write_text(content, encoding="utf-8")
        return self.workspace.relative_to_workspace(safe_path.relative_to(self.workspace.root))

    def append_file(self, path: str | Path, content: str) -> str:
        """Append content to an allowed file in the workspace."""
        safe_path = self._resolve_allowed_file(path)
        existing_size = safe_path.stat().st_size if safe_path.exists() else 0
        new_size = existing_size + len(content.encode("utf-8"))
        if new_size > self.max_file_size_bytes:
            raise FileTooLargeError("File would exceed the maximum allowed size.")
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        with safe_path.open("a", encoding="utf-8") as file:
            file.write(content)
        return self.workspace.relative_to_workspace(safe_path.relative_to(self.workspace.root))

    def list_files(self, path: str | Path = ".") -> list[str]:
        """List files below a safe workspace directory."""
        safe_path = self.workspace.resolve_safe_path(path)
        if not safe_path.exists():
            return []
        if safe_path.is_file():
            self._validate_extension(safe_path)
            relative_path = safe_path.relative_to(self.workspace.root)
            return [self.workspace.relative_to_workspace(relative_path)]

        files: list[str] = []
        for item in safe_path.rglob("*"):
            if item.is_file() and item.suffix in self.allowed_extensions:
                files.append(
                    self.workspace.relative_to_workspace(item.relative_to(self.workspace.root))
                )
        return sorted(files)

    def exists(self, path: str | Path) -> bool:
        """Return whether a safe workspace path exists."""
        safe_path = self._resolve_allowed_file(path)
        return safe_path.exists()

    def _resolve_allowed_file(self, path: str | Path) -> Path:
        try:
            safe_path = self.workspace.resolve_safe_path(path)
        except WorkspaceSecurityError as error:
            raise FileToolError(str(error)) from error
        self._validate_extension(safe_path)
        return safe_path

    def _validate_extension(self, path: Path) -> None:
        if path.suffix not in self.allowed_extensions:
            raise UnsupportedFileTypeError(f"Extension not allowed: {path.suffix}")

    def _validate_content_size(self, content: str) -> None:
        if len(content.encode("utf-8")) > self.max_file_size_bytes:
            raise FileTooLargeError("Content exceeds the maximum allowed size.")
