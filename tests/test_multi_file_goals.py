from pathlib import Path

from multi_agent_lab.agents.reviewer_agent import ReviewerAgent
from multi_agent_lab.core.file_awareness import FileAwarenessService
from multi_agent_lab.core.workspace_manager import WorkspaceManager
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


async def test_flask_todo_mock_generates_stable_baseline(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    runtime = AgentRuntime(
        "Crea una pequena API Flask TODO con tests basicos",
        workspace_path=str(workspace),
        database_url=f"sqlite:///{tmp_path / 'runtime.db'}",
        use_mock_llm=True,
        timeout_seconds=10,
        allow_execution=True,
    )

    summary = await runtime.run()

    app = (workspace / "app.py").read_text(encoding="utf-8")
    requirements = (workspace / "requirements.txt").read_text(encoding="utf-8")
    tests = (workspace / "tests" / "test_app.py").read_text(encoding="utf-8")
    assert summary.status == "completed"
    assert sorted(summary.files_created) == [
        "README.md",
        "app.py",
        "requirements.txt",
        "tests/test_app.py",
    ]
    assert "app = Flask(__name__)" in app
    assert "todos = []" in app
    assert "@app.get('/todos')" in app
    assert "@app.post('/todos')" in app
    assert "@app.get('/todos/<int:todo_id>')" in app
    assert "@app.delete('/todos/<int:todo_id>')" in app
    assert requirements == "Flask\npytest\n"
    assert "from app import app" in tests
    assert "client = app.test_client()" in tests
    assert "db" not in tests
    assert "Todo" not in tests
    assert "create_app" not in tests


async def test_flask_todo_baseline_pytest_passes_with_execution(tmp_path: Path) -> None:
    runtime = AgentRuntime(
        "Crea una pequena API Flask TODO con tests basicos",
        workspace_path=str(tmp_path / "workspace"),
        database_url=f"sqlite:///{tmp_path / 'runtime.db'}",
        use_mock_llm=True,
        timeout_seconds=10,
        allow_execution=True,
    )

    summary = await runtime.run()

    assert summary.status == "completed"
    assert summary.execution_success_count == 1
    assert summary.execution_failure_count == 0


async def test_python_cli_mock_generates_stable_baseline(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    runtime = AgentRuntime(
        "Crea una CLI Python para gestionar tareas con tests basicos",
        workspace_path=str(workspace),
        database_url=f"sqlite:///{tmp_path / 'runtime.db'}",
        use_mock_llm=True,
        timeout_seconds=10,
        allow_execution=True,
    )

    summary = await runtime.run()

    cli = (workspace / "task_cli.py").read_text(encoding="utf-8")
    requirements = (workspace / "requirements.txt").read_text(encoding="utf-8")
    tests = (workspace / "tests" / "test_task_cli.py").read_text(encoding="utf-8")
    readme = (workspace / "README.md").read_text(encoding="utf-8")
    assert summary.status == "completed"
    assert sorted(summary.files_created) == [
        "README.md",
        "requirements.txt",
        "task_cli.py",
        "tests/test_task_cli.py",
    ]
    assert "import argparse" in cli
    assert "def add_task" in cli
    assert "def list_tasks" in cli
    assert "def mark_done" in cli
    assert "add_parser = subparsers.add_parser('add')" in cli
    assert "subparsers.add_parser('list')" in cli
    assert "done_parser = subparsers.add_parser('done')" in cli
    assert requirements == "pytest\n"
    assert "from task_cli import add_task, list_tasks, mark_done" in tests
    assert "import db" not in tests
    assert "Todo" not in tests
    assert "create_app" not in tests
    assert "CLI Python" in readme or "Task CLI" in readme


async def test_python_cli_baseline_pytest_passes_with_execution(tmp_path: Path) -> None:
    runtime = AgentRuntime(
        "Crea una CLI Python para gestionar tareas con tests basicos",
        workspace_path=str(tmp_path / "workspace"),
        database_url=f"sqlite:///{tmp_path / 'runtime.db'}",
        use_mock_llm=True,
        timeout_seconds=10,
        allow_execution=True,
    )

    summary = await runtime.run()

    assert summary.status == "completed"
    assert summary.execution_success_count == 1
    assert summary.execution_failure_count == 0


async def test_python_cli_execution_targets_cli_tests(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    tests_dir = workspace / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test_app.py").write_text(
        "def test_unrelated_failure():\n    assert False\n",
        encoding="utf-8",
    )
    runtime = AgentRuntime(
        "Crea una CLI Python para gestionar tareas con tests basicos",
        workspace_path=str(workspace),
        database_url=f"sqlite:///{tmp_path / 'runtime.db'}",
        use_mock_llm=True,
        timeout_seconds=10,
        allow_execution=True,
    )

    summary = await runtime.run()

    assert summary.status == "completed"
    assert summary.execution_success_count == 1
    assert summary.execution_failure_count == 0


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

    assert awareness.read_relevant_files(["../secret.txt"]) == {}
    assert awareness.invalid_paths_detected == 1
    assert awareness.invalid_paths_ignored == 1
