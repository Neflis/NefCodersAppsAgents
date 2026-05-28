from pathlib import Path

from multi_agent_lab.agents.reviewer_agent import ReviewerAgent
from multi_agent_lab.core.file_awareness import FileAwarenessService
from multi_agent_lab.core.workspace_manager import WorkspaceManager, WorkspaceSecurityError
from multi_agent_lab.runtime import AgentRuntime
from multi_agent_lab.tools.file_tool import FileTool


async def test_flask_goal_creates_multiple_files(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    runtime = AgentRuntime(
        "Crea una pequena API Flask TODO",
        workspace_path=str(workspace),
        database_url=f"sqlite:///{tmp_path / 'runtime.db'}",
        use_mock_llm=True,
        timeout_seconds=5,
    )

    summary = await runtime.run()

    assert summary.status == "completed"
    assert sorted(summary.files_created) == ["README.md", "app.py", "requirements.txt"]


async def test_flask_files_are_coherent(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    runtime = AgentRuntime(
        "Crea una pequena API Flask TODO",
        workspace_path=str(workspace),
        database_url=f"sqlite:///{tmp_path / 'runtime.db'}",
        use_mock_llm=True,
        timeout_seconds=5,
    )

    await runtime.run()

    app = (workspace / "app.py").read_text(encoding="utf-8")
    requirements = (workspace / "requirements.txt").read_text(encoding="utf-8")
    readme = (workspace / "README.md").read_text(encoding="utf-8")
    assert "from flask import" in app
    assert "Flask" in requirements
    assert "Flask TODO API" in readme
    assert "/todos" in readme


def test_reviewer_detects_flask_import_without_requirement() -> None:
    reviewer = ReviewerAgent("reviewer", None, None)  # type: ignore[arg-type]

    feedback = reviewer._project_feedback(
        {
            "app.py": "from flask import Flask\n",
            "requirements.txt": "",
            "README.md": "# Flask TODO API\n\nEndpoint /todos\n",
        }
    )

    assert "requirements.txt debe incluir Flask." in feedback


def test_file_awareness_does_not_read_outside_workspace(tmp_path: Path) -> None:
    workspace = WorkspaceManager(tmp_path / "workspace")
    awareness = FileAwarenessService(FileTool(workspace))

    try:
        awareness.read_relevant_files(["../secret.txt"])
    except WorkspaceSecurityError:
        pass
    else:
        raise AssertionError("Path traversal should be blocked.")
