from pathlib import Path

from multi_agent_lab.agents.coder_agent import CoderAgent
from multi_agent_lab.agents.file_agent import FileAgent
from multi_agent_lab.agents.reviewer_agent import ReviewerAgent
from multi_agent_lab.core.capability import Capability
from multi_agent_lab.core.code_content_sanitizer import CodeContentSanitizer
from multi_agent_lab.core.failure_analysis import FailureAnalysisService, FixStrategy
from multi_agent_lab.core.file_awareness import FileAwarenessService
from multi_agent_lab.core.fix_target_guard import FixTargetGuard
from multi_agent_lab.core.message import EventType, Message
from multi_agent_lab.core.message_bus import MessageBus
from multi_agent_lab.core.task_graph_store import TaskGraphStore
from multi_agent_lab.core.workspace_manager import WorkspaceManager
from multi_agent_lab.llm.context_builder import AgentContextBuilder
from multi_agent_lab.tools.command_tool import CommandTool
from multi_agent_lab.tools.file_tool import FileTool


def test_sanitizer_removes_python_fences() -> None:
    sanitizer = CodeContentSanitizer()

    content = sanitizer.normalize_code_content("app.py", "```python\nprint('ok')\n```\n")

    assert content == "print('ok')\n"
    assert sanitizer.sanitized_files_count == 1


def test_sanitizer_keeps_markdown_fences() -> None:
    sanitizer = CodeContentSanitizer()
    markdown = "```python\nprint('ok')\n```\n"

    assert sanitizer.normalize_code_content("README.md", markdown) == markdown
    assert sanitizer.sanitized_files_count == 0


async def test_file_agent_does_not_write_python_fences(tmp_path: Path) -> None:
    bus = MessageBus()
    workspace = WorkspaceManager(tmp_path / "workspace")
    sanitizer = CodeContentSanitizer()
    agent = FileAgent(
        "file_agent",
        bus,
        FileTool(workspace),
        TaskGraphStore(),
        content_sanitizer=sanitizer,
    )

    await agent.handle_message(
        Message(
            sender="coder",
            type=EventType.CODE_PROPOSED,
            content={
                "task_id": "task",
                "path": "app.py",
                "content": "```python\nprint('ok')\n```\n",
            },
            correlation_id="corr",
        )
    )

    assert (tmp_path / "workspace" / "app.py").read_text(encoding="utf-8") == "print('ok')\n"
    assert sanitizer.sanitized_files_count == 1


async def test_file_agent_rejects_local_import_fix_to_requirements(tmp_path: Path) -> None:
    bus = MessageBus()
    failed = await bus.subscribe(EventType.FIX_FAILED)
    workspace = WorkspaceManager(tmp_path / "workspace")
    guard = FixTargetGuard()
    agent = FileAgent(
        "file_agent",
        bus,
        FileTool(workspace),
        TaskGraphStore(),
        fix_target_guard=guard,
    )

    await agent.handle_message(
        Message(
            sender="coder",
            type=EventType.FIX_PROPOSED,
            content={
                "task_id": "fix-task",
                "path": "requirements.txt",
                "content": "Flask>=3.0\n",
                "fix_strategy": FixStrategy.FIX_LOCAL_MODULE_IMPORT.value,
                "failure_context": {"failure_type": "LocalModuleNotFoundError"},
            },
            correlation_id="corr",
        )
    )

    event = await failed.get()
    assert event.content["error"] == "wrong_target_fix"
    assert guard.wrong_target_fix_count == 1
    assert not (workspace.root / "requirements.txt").exists()


def test_failure_analysis_detects_markdown_fence_syntax_error() -> None:
    service = FailureAnalysisService()
    stderr = 'File "workspace/tests/test_app.py", line 3\n```python\n^\nSyntaxError: invalid syntax'

    context = service.parse_pytest_output("", stderr)
    strategy = CoderAgent("coder", None)._select_fix_strategy(  # type: ignore[arg-type]
        context.to_dict(),
        "tests/test_app.py",
    )

    assert context.failure_type == "MarkdownFenceSyntaxError"
    assert strategy == FixStrategy.STRIP_MARKDOWN_FENCES


def test_module_not_found_app_uses_local_import_strategy() -> None:
    context = FailureAnalysisService().parse_pytest_output(
        "FAILED tests/test_app.py::test_health",
        "ModuleNotFoundError: No module named 'app'",
    )
    strategy = CoderAgent("coder", None)._select_fix_strategy(  # type: ignore[arg-type]
        context.to_dict(),
        "tests/test_app.py",
    )

    assert context.failure_type == "LocalModuleNotFoundError"
    assert context.suspected_files == ["tests/test_app.py", "app.py"]
    assert strategy == FixStrategy.FIX_LOCAL_MODULE_IMPORT


def test_module_not_found_external_uses_requirements() -> None:
    for module_name in ("flask", "sqlalchemy"):
        context = FailureAnalysisService().parse_pytest_output(
            "FAILED tests/test_app.py::test_health",
            f"ModuleNotFoundError: No module named '{module_name}'",
        )
        strategy = CoderAgent("coder", None)._select_fix_strategy(  # type: ignore[arg-type]
            context.to_dict(),
            "requirements.txt",
        )

        assert context.failure_type == "ModuleNotFoundError"
        assert context.suspected_files[0] == "requirements.txt"
        assert strategy == FixStrategy.ADD_MISSING_DEPENDENCY


def test_reviewer_rejects_python_markdown_fences() -> None:
    reviewer = ReviewerAgent("reviewer", None, TaskGraphStore())  # type: ignore[arg-type]

    assert not reviewer._is_valid_content("app.py", "```python\nprint('bad')\n```\n")


def test_local_import_fix_removes_create_app_and_db_imports(tmp_path: Path) -> None:
    workspace = WorkspaceManager(tmp_path / "workspace")
    file_tool = FileTool(workspace)
    file_tool.write_file("app.py", "from flask import Flask\napp = Flask(__name__)\n")
    file_tool.write_file(
        "tests/test_app.py",
        "from app import create_app, db\n\n\ndef test_health():\n"
        "    client = create_app().test_client()\n    assert client.get('/todos').status_code\n",
    )
    coder = CoderAgent(
        "coder",
        None,  # type: ignore[arg-type]
        context_builder=AgentContextBuilder(
            file_tool=file_tool,
            file_awareness=FileAwarenessService(file_tool),
        ),
    )

    fixed = coder._direct_local_import_fix("tests/test_app.py")

    assert "from app import app" in fixed
    assert "db" not in fixed
    assert "create_app" not in fixed


def test_local_import_fix_keeps_create_app_when_defined(tmp_path: Path) -> None:
    workspace = WorkspaceManager(tmp_path / "workspace")
    file_tool = FileTool(workspace)
    file_tool.write_file(
        "app.py",
        "from flask import Flask\napp = Flask(__name__)\n\ndef create_app():\n    return app\n",
    )
    file_tool.write_file(
        "tests/test_app.py",
        "from app import create_app, db\n\n\ndef test_health():\n"
        "    client = create_app().test_client()\n    assert client.get('/todos').status_code\n",
    )
    coder = CoderAgent(
        "coder",
        None,  # type: ignore[arg-type]
        context_builder=AgentContextBuilder(
            file_tool=file_tool,
            file_awareness=FileAwarenessService(file_tool),
        ),
    )

    fixed = coder._direct_local_import_fix("tests/test_app.py")

    assert "from app import create_app" in fixed
    assert "db" not in fixed
    assert "create_app().test_client()" in fixed


async def test_auto_fix_removes_fences_and_pytest_can_continue(tmp_path: Path) -> None:
    workspace = WorkspaceManager(tmp_path / "workspace")
    file_tool = FileTool(workspace)
    file_tool.write_file("app.py", "def value():\n    return 1\n")
    file_tool.write_file(
        "tests/test_app.py",
        "```python\nfrom app import value\n\n\ndef test_value():\n    assert value() == 1\n```\n",
    )
    context_builder = AgentContextBuilder(
        file_tool=file_tool,
        file_awareness=FileAwarenessService(file_tool),
    )
    bus = MessageBus()
    proposed = await bus.subscribe(EventType.FIX_PROPOSED)
    coder = CoderAgent("coder", bus, context_builder=context_builder)
    message = Message(
        sender="coordinator",
        type=EventType.FIX_REQUESTED,
        content={
            "task_id": "fix-task",
            "required_capability": Capability.CODING.value,
            "status": "READY",
            "payload": {
                "type": "fix",
                "path": "tests/test_app.py",
                "failure": {
                    "stderr": (
                        'File "workspace/tests/test_app.py", line 1\n'
                        "```python\n^\nSyntaxError: invalid syntax"
                    )
                },
                "failure_context": {
                    "failure_type": "MarkdownFenceSyntaxError",
                    "traceback": "```python\nSyntaxError: invalid syntax",
                    "suspected_files": ["tests/test_app.py"],
                },
            },
        },
        correlation_id="corr",
    )

    await coder.handle_message(message)
    event = await proposed.get()
    file_tool.write_file("tests/test_app.py", str(event.content["content"]))

    assert "```" not in (workspace.root / "tests" / "test_app.py").read_text(encoding="utf-8")
    assert CommandTool(workspace).run_pytest().success
