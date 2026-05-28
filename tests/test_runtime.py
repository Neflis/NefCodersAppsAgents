from pathlib import Path

from multi_agent_lab.runtime import AgentRuntime


async def test_runtime_mock_completes(tmp_path: Path) -> None:
    runtime = AgentRuntime(
        "Crear README para app TODO",
        workspace_path=str(tmp_path / "workspace"),
        database_url=f"sqlite:///{tmp_path / 'runtime.db'}",
        use_mock_llm=True,
        timeout_seconds=3,
    )

    summary = await runtime.run()

    assert summary.status == "completed"
    assert summary.tasks_completed == 4
    assert summary.tasks_failed == 0
    assert summary.files_created == ["README.md"]
    assert (tmp_path / "workspace" / "README.md").exists()


async def test_runtime_timeout_stops_workflow(tmp_path: Path) -> None:
    runtime = AgentRuntime(
        "Crear README para app TODO",
        workspace_path=str(tmp_path / "workspace"),
        database_url=f"sqlite:///{tmp_path / 'runtime.db'}",
        use_mock_llm=True,
        timeout_seconds=0.0001,
    )

    summary = await runtime.run()

    assert summary.status == "timeout"
    assert summary.terminal_event == "WORKFLOW_TIMEOUT"


async def test_runtime_does_not_write_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "sandbox"
    outside = tmp_path / "README.md"
    runtime = AgentRuntime(
        "Crear README para app TODO",
        workspace_path=str(workspace),
        database_url=f"sqlite:///{tmp_path / 'runtime.db'}",
        use_mock_llm=True,
        timeout_seconds=3,
    )

    summary = await runtime.run()

    assert summary.status == "completed"
    assert (workspace / "README.md").exists()
    assert not outside.exists()
