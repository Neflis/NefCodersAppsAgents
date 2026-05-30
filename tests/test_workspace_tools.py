from pathlib import Path

import pytest

from multi_agent_lab.core.workspace_manager import WorkspaceManager, WorkspaceSecurityError
from multi_agent_lab.tools.file_tool import (
    FileTool,
    FileTooLargeError,
    UnsupportedFileTypeError,
)


def test_workspace_manager_blocks_parent_traversal(tmp_path: Path) -> None:
    manager = WorkspaceManager(tmp_path / "workspace")

    with pytest.raises(WorkspaceSecurityError):
        manager.resolve_safe_path("../../secret.txt")


def test_workspace_manager_blocks_absolute_paths(tmp_path: Path) -> None:
    manager = WorkspaceManager(tmp_path / "workspace")

    with pytest.raises(WorkspaceSecurityError):
        manager.resolve_safe_path(tmp_path / "outside.md")


def test_file_tool_writes_and_reads_allowed_file(tmp_path: Path) -> None:
    tool = FileTool(WorkspaceManager(tmp_path / "workspace"))

    written_path = tool.write_file("README.md", "# Example\n")
    content = tool.read_file("README.md")

    assert written_path == "README.md"
    assert content == "# Example\n"
    assert tool.exists("README.md")
    assert (tmp_path / "workspace" / "README.md").exists()


def test_file_tool_appends_allowed_file(tmp_path: Path) -> None:
    tool = FileTool(WorkspaceManager(tmp_path / "workspace"))

    tool.write_file("notes.txt", "hello")
    tool.append_file("notes.txt", " world")

    assert tool.read_file("notes.txt") == "hello world"


def test_file_tool_allows_yaml_files(tmp_path: Path) -> None:
    tool = FileTool(WorkspaceManager(tmp_path / "workspace"))

    tool.write_file("config.yaml", "name: demo\n")
    tool.write_file("config.yml", "name: demo\n")

    assert tool.exists("config.yaml")
    assert tool.exists("config.yml")


def test_file_tool_allows_spring_boot_file_types(tmp_path: Path) -> None:
    tool = FileTool(WorkspaceManager(tmp_path / "workspace"))

    tool.write_file("pom.xml", "<project></project>\n")
    tool.write_file(
        "src/main/java/com/example/demo/DemoApplication.java",
        "package com.example.demo;\n",
    )

    assert tool.exists("pom.xml")
    assert tool.exists("src/main/java/com/example/demo/DemoApplication.java")


def test_file_tool_allows_angular_file_types(tmp_path: Path) -> None:
    tool = FileTool(WorkspaceManager(tmp_path / "workspace"))

    tool.write_file("src/main.ts", "export {};\n")
    tool.write_file("src/app/app.component.html", "<h1>Angular Works</h1>\n")

    assert tool.exists("src/main.ts")
    assert tool.exists("src/app/app.component.html")


def test_file_tool_blocks_disallowed_extension(tmp_path: Path) -> None:
    tool = FileTool(WorkspaceManager(tmp_path / "workspace"))

    with pytest.raises(UnsupportedFileTypeError):
        tool.write_file("secret.exe", "blocked")


def test_file_tool_blocks_large_content(tmp_path: Path) -> None:
    tool = FileTool(WorkspaceManager(tmp_path / "workspace"))

    with pytest.raises(FileTooLargeError):
        tool.write_file("large.txt", "x" * (1024 * 1024 + 1))


def test_file_tool_lists_allowed_files(tmp_path: Path) -> None:
    tool = FileTool(WorkspaceManager(tmp_path / "workspace"))
    tool.write_file("README.md", "# Demo\n")
    tool.write_file("src/app.py", "print('demo')\n")

    assert tool.list_files(".") == ["README.md", "src/app.py"]
