from pathlib import Path

import pytest

from multi_agent_lab.core.workspace_manager import WorkspaceManager
from multi_agent_lab.tools.patch_tool import PatchTool, PatchToolError, PatchValidationError


def test_valid_patch_modifies_file(tmp_path: Path) -> None:
    workspace = WorkspaceManager(tmp_path / "workspace")
    target = workspace.root / "README.md"
    target.write_text("hello world\n", encoding="utf-8")
    tool = PatchTool(workspace)

    result = tool.apply_search_replace("README.md", "hello", "hola")

    assert target.read_text(encoding="utf-8") == "hola world\n"
    assert result.path == "README.md"
    assert result.diff_summary == "README.md: +1 -1"
    assert tool.backup_for("README.md") == "hello world\n"


def test_patch_with_missing_search_fails(tmp_path: Path) -> None:
    workspace = WorkspaceManager(tmp_path / "workspace")
    (workspace.root / "README.md").write_text("hello world\n", encoding="utf-8")
    tool = PatchTool(workspace)

    with pytest.raises(PatchValidationError, match="not found"):
        tool.apply_search_replace("README.md", "missing", "hola")


def test_patch_with_ambiguous_search_fails(tmp_path: Path) -> None:
    workspace = WorkspaceManager(tmp_path / "workspace")
    (workspace.root / "README.md").write_text("hello hello\n", encoding="utf-8")
    tool = PatchTool(workspace)

    with pytest.raises(PatchValidationError, match="ambiguous"):
        tool.apply_search_replace("README.md", "hello", "hola")


def test_patch_outside_workspace_fails(tmp_path: Path) -> None:
    workspace = WorkspaceManager(tmp_path / "workspace")
    (workspace.root / "README.md").write_text("hello\n", encoding="utf-8")
    tool = PatchTool(workspace)

    with pytest.raises(PatchToolError, match="Path traversal"):
        tool.apply_search_replace("../README.md", "hello", "hola")


def test_patch_does_not_create_new_files(tmp_path: Path) -> None:
    workspace = WorkspaceManager(tmp_path / "workspace")
    tool = PatchTool(workspace)

    with pytest.raises(PatchValidationError, match="cannot create"):
        tool.apply_search_replace("README.md", "hello", "hola")


def test_patch_rejects_binary_files(tmp_path: Path) -> None:
    workspace = WorkspaceManager(tmp_path / "workspace")
    (workspace.root / "data.txt").write_bytes(b"hello\x00world")
    tool = PatchTool(workspace)

    with pytest.raises(PatchValidationError, match="Binary"):
        tool.apply_search_replace("data.txt", "hello", "hola")
