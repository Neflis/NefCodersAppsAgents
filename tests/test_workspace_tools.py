from pathlib import Path

import pytest

from multi_agent_lab.core.workspace_manager import WorkspaceManager, WorkspaceSecurityError
from multi_agent_lab.tools.file_tool import FileTool, FileToolError


def test_workspace_manager_blocks_path_traversal(tmp_path: Path) -> None:
    manager = WorkspaceManager(tmp_path / "workspace")

    with pytest.raises(WorkspaceSecurityError):
        manager.resolve_safe_path("../outside.md")


def test_workspace_manager_blocks_absolute_paths(tmp_path: Path) -> None:
    manager = WorkspaceManager(tmp_path / "workspace")

    with pytest.raises(WorkspaceSecurityError):
        manager.resolve_safe_path(tmp_path / "outside.md")


def test_file_tool_writes_and_reads_allowed_file(tmp_path: Path) -> None:
    tool = FileTool(WorkspaceManager(tmp_path / "workspace"))

    written_path = tool.write_file("docs/example.md", "# Example\n")
    content = tool.read_file("docs/example.md")

    assert written_path == "docs/example.md"
    assert content == "# Example\n"
    assert (tmp_path / "workspace" / "docs" / "example.md").exists()


def test_file_tool_blocks_disallowed_extension(tmp_path: Path) -> None:
    tool = FileTool(WorkspaceManager(tmp_path / "workspace"))

    with pytest.raises(FileToolError):
        tool.write_file("secret.exe", "blocked")


def test_file_tool_blocks_large_content(tmp_path: Path) -> None:
    tool = FileTool(WorkspaceManager(tmp_path / "workspace"), max_file_size_bytes=4)

    with pytest.raises(FileToolError):
        tool.write_file("large.txt", "too large")


def test_file_tool_lists_allowed_files(tmp_path: Path) -> None:
    tool = FileTool(WorkspaceManager(tmp_path / "workspace"))
    tool.write_file("README.md", "# Demo\n")
    tool.write_file("src/app.py", "print('demo')\n")

    assert tool.list_files(".") == ["README.md", "src/app.py"]
