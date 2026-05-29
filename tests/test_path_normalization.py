from pathlib import Path

import pytest

from multi_agent_lab.agents.coder_agent import CoderAgent
from multi_agent_lab.core.capability import Capability
from multi_agent_lab.core.failure_analysis import FailureAnalysisService
from multi_agent_lab.core.file_awareness import FileAwarenessService
from multi_agent_lab.core.file_path_normalizer import FilePathNormalizer
from multi_agent_lab.core.message import EventType, Message
from multi_agent_lab.core.message_bus import MessageBus
from multi_agent_lab.core.workspace_manager import WorkspaceManager, WorkspaceSecurityError
from multi_agent_lab.llm.context_builder import AgentContextBuilder
from multi_agent_lab.tools.file_tool import FileTool


class FakeFixLLM:
    async def generate_json(self, prompt: str, required_fields=None):  # noqa: ANN001
        return {"content": "# fixed\n", "reasoning_summary": "ok"}

    def record_fallback(self) -> None:
        pass


def test_normalizer_converts_workspace_absolute_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    normalizer = FilePathNormalizer(workspace)

    assert normalizer.normalize_workspace_path(workspace / "app.py") == "app.py"


def test_normalizer_rejects_external_absolute_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    normalizer = FilePathNormalizer(workspace)

    with pytest.raises(WorkspaceSecurityError):
        normalizer.normalize_workspace_path(tmp_path / "outside.py")


def test_failure_analysis_converts_valid_traceback_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = FailureAnalysisService(FilePathNormalizer(workspace))
    output = f'Traceback (most recent call last):\n  File "{workspace / "app.py"}", line 3\n'

    context = service.parse_pytest_output(output, "NameError: name 'x' is not defined")

    assert context.suspected_files == ["app.py"]


def test_failure_analysis_ignores_external_traceback_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = FailureAnalysisService(FilePathNormalizer(workspace))
    output = f'Traceback (most recent call last):\n  File "{tmp_path / "outside.py"}", line 3\n'

    context = service.parse_pytest_output(output, "NameError: name 'x' is not defined")

    assert context.suspected_files == []


def test_context_builder_ignores_invalid_workspace_path(tmp_path: Path) -> None:
    workspace = WorkspaceManager(tmp_path / "workspace")
    file_tool = FileTool(workspace)
    awareness = FileAwarenessService(file_tool)
    builder = AgentContextBuilder(file_tool=file_tool, file_awareness=awareness)
    message = Message(sender="test", type=EventType.FIX_REQUESTED, content={})

    context = builder.build(message, workspace_paths=[str(tmp_path / "outside.py")])

    assert context["workspace_files"] == {}
    assert awareness.invalid_paths_ignored == 1


async def test_coder_fix_continues_after_invalid_context_path(tmp_path: Path) -> None:
    workspace = WorkspaceManager(tmp_path / "workspace")
    file_tool = FileTool(workspace)
    context_builder = AgentContextBuilder(
        file_tool=file_tool,
        file_awareness=FileAwarenessService(file_tool),
    )
    bus = MessageBus()
    proposed = await bus.subscribe(EventType.FIX_PROPOSED)
    coder = CoderAgent(
        "coder",
        bus,
        ollama_client=FakeFixLLM(),  # type: ignore[arg-type]
        context_builder=context_builder,
    )
    message = Message(
        sender="coordinator",
        type=EventType.FIX_REQUESTED,
        content={
            "task_id": "fix-task",
            "required_capability": Capability.CODING.value,
            "status": "READY",
            "payload": {
                "type": "fix",
                "path": "README.md",
                "failure": {"stderr": "AssertionError"},
                "failure_context": {
                    "failure_type": "AssertionError",
                    "suspected_files": [str(tmp_path / "outside.py")],
                },
            },
        },
        correlation_id="corr",
    )

    await coder.handle_message(message)

    event = await proposed.get()
    assert event.content["content"] == "# fixed\n"
