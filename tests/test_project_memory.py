from pathlib import Path

from multi_agent_lab.core.event_noise import EventNoiseReducer
from multi_agent_lab.core.message import EventType, Message
from multi_agent_lab.core.project_memory_service import ProjectMemoryService
from multi_agent_lab.core.sqlite_store import SQLiteStore
from multi_agent_lab.llm.context_builder import AgentContextBuilder
from multi_agent_lab.runtime import AgentRuntime


def test_project_memory_is_created(tmp_path: Path) -> None:
    store = SQLiteStore(f"sqlite:///{tmp_path / 'memory.db'}")
    service = ProjectMemoryService(store)

    memory = service.get_memory("corr-1")

    assert memory.correlation_id == "corr-1"
    assert store.load_project_memory("corr-1") is not None
    store.close()


def test_project_memory_updates_with_file_written(tmp_path: Path) -> None:
    store = SQLiteStore(f"sqlite:///{tmp_path / 'memory.db'}")
    service = ProjectMemoryService(store)

    service.update_from_event(
        Message(
            sender="file_agent",
            type=EventType.FILE_WRITTEN,
            content={"path": "app.py"},
            correlation_id="corr-1",
        )
    )

    memory = service.get_memory("corr-1")
    assert memory.files_created == ["app.py"]
    assert store.load_project_memory("corr-1")["files_created"] == ["app.py"]
    store.close()


def test_project_memory_updates_goal_summary(tmp_path: Path) -> None:
    service = ProjectMemoryService(SQLiteStore(f"sqlite:///{tmp_path / 'memory.db'}"))

    service.update_from_event(
        Message(
            sender="planner",
            type=EventType.GOAL_DECOMPOSED,
            content={"goal": "Crea una pequena API Flask TODO"},
            correlation_id="corr-1",
        )
    )

    memory = service.get_memory("corr-1")
    assert memory.goal_summary == "Crea una pequena API Flask TODO"
    assert memory.detected_framework == "Flask"


def test_project_memory_stores_reviewer_feedback(tmp_path: Path) -> None:
    service = ProjectMemoryService(SQLiteStore(f"sqlite:///{tmp_path / 'memory.db'}"))

    service.update_from_event(
        Message(
            sender="reviewer",
            type=EventType.PROJECT_REVIEW_REJECTED,
            content={"feedback": ["requirements.txt debe incluir Flask."]},
            correlation_id="corr-1",
        )
    )

    memory = service.get_memory("corr-1")
    assert "requirements.txt debe incluir Flask." in memory.reviewer_feedback
    assert "requirements.txt debe incluir Flask." in memory.known_errors


def test_project_memory_does_not_repeat_fix_hash(tmp_path: Path) -> None:
    service = ProjectMemoryService(SQLiteStore(f"sqlite:///{tmp_path / 'memory.db'}"))

    message = Message(
        sender="coder",
        type=EventType.FIX_PROPOSED,
        content={"content_hash": "abc123"},
        correlation_id="corr-1",
    )
    service.update_from_event(message)
    service.update_from_event(message)

    memory = service.get_memory("corr-1")
    assert memory.proposed_fix_hashes == ["abc123"]
    assert service.has_fix_hash("corr-1", "abc123")


def test_project_memory_records_app_exports(tmp_path: Path) -> None:
    service = ProjectMemoryService(SQLiteStore(f"sqlite:///{tmp_path / 'memory.db'}"))

    service.update_from_event(
        Message(
            sender="coder",
            type=EventType.CODE_PROPOSED,
            content={
                "path": "app.py",
                "content": "from flask import Flask\napp = Flask(__name__)\n"
                "def create_app():\n    return app\n",
            },
            correlation_id="corr-1",
        )
    )

    memory = service.get_memory("corr-1")
    assert "app" in memory.exported_symbols
    assert "create_app" in memory.exported_symbols


def test_context_builder_includes_project_memory_summary(tmp_path: Path) -> None:
    service = ProjectMemoryService(SQLiteStore(f"sqlite:///{tmp_path / 'memory.db'}"))
    service.add_file("corr-1", "app.py")
    service.add_decision("corr-1", "Use Flask route decorators.")
    builder = AgentContextBuilder(project_memory=service)

    context = builder.build(
        Message(
            sender="tester",
            type=EventType.TEST_PASSED,
            content={},
            correlation_id="corr-1",
        )
    )

    assert "app.py" in context["project_memory"]
    assert "Use Flask route decorators." in context["project_memory"]


def test_context_builder_respects_max_chars(tmp_path: Path) -> None:
    service = ProjectMemoryService(SQLiteStore(f"sqlite:///{tmp_path / 'memory.db'}"))
    for index in range(50):
        service.add_decision("corr-1", f"Decision {index}: {'x' * 40}")
    builder = AgentContextBuilder(project_memory=service, max_context_chars=500)

    rendered = builder.build_json(
        Message(
            sender="planner",
            type=EventType.GOAL_SUBMITTED,
            content={"goal": "Crea una pequena API Flask TODO"},
            correlation_id="corr-1",
        )
    )

    assert len(rendered) <= 500


def test_event_noise_reducer_groups_repeated_graph_updates() -> None:
    reducer = EventNoiseReducer(verbose=False)
    first = Message(sender="coordinator", type=EventType.TASK_GRAPH_UPDATED, content={})
    second = Message(sender="coordinator", type=EventType.TASK_GRAPH_UPDATED, content={})

    reducer.record(first)
    reducer.record(second)

    assert reducer.should_log(first)
    assert not reducer.should_log(second)
    assert reducer.summary()["event_counts"]["TASK_GRAPH_UPDATED"] == 2
    assert reducer.summary()["suppressed_counts"]["TASK_GRAPH_UPDATED"] == 1


async def test_demo_mock_still_creates_multiple_files(tmp_path: Path) -> None:
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
    assert "event_counts" in summary.event_summary


async def test_demo_ollama_mode_handles_empty_memory_when_unavailable(tmp_path: Path) -> None:
    runtime = AgentRuntime(
        "Crea una pequena API Flask TODO",
        workspace_path=str(tmp_path / "workspace"),
        database_url=f"sqlite:///{tmp_path / 'runtime.db'}",
        use_mock_llm=False,
        timeout_seconds=0.01,
    )

    summary = await runtime.run()

    assert summary.status in {"completed", "halted", "timeout"}
    assert summary.event_summary["event_counts"]
