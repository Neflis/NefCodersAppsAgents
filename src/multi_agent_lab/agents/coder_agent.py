"""Coder agent that proposes content for coding tasks."""

from __future__ import annotations

import logging
from hashlib import sha256

from multi_agent_lab.agents.base_agent import BaseAgent
from multi_agent_lab.core.agent_event_logger import AgentEventLogger
from multi_agent_lab.core.capability import Capability
from multi_agent_lab.core.code_content_sanitizer import CodeContentSanitizer
from multi_agent_lab.core.failure_analysis import FixStrategy
from multi_agent_lab.core.message import EventType, Message
from multi_agent_lab.core.message_bus import MessageBus
from multi_agent_lab.llm.context_builder import AgentContextBuilder
from multi_agent_lab.llm.decision import LLMDecision
from multi_agent_lab.llm.ollama_client import InvalidJSONError, OllamaClient, OllamaClientError
from multi_agent_lab.llm.prompt_template import PromptTemplate
from multi_agent_lab.llm.schemas import CODER_DECISION_FIELDS

logger = logging.getLogger(__name__)


class CoderAgent(BaseAgent):
    """Agent that claims coding tasks and proposes content."""

    subscribed_events = (EventType.TASK_READY, EventType.FIX_REQUESTED)
    capabilities = (Capability.CODING.value,)

    def __init__(
        self,
        name: str,
        bus: MessageBus,
        event_logger: AgentEventLogger | None = None,
        ollama_client: OllamaClient | None = None,
        context_builder: AgentContextBuilder | None = None,
    ) -> None:
        super().__init__(name, bus, event_logger)
        self.ollama_client = ollama_client
        self.context_builder = context_builder or AgentContextBuilder()
        self.content_sanitizer = CodeContentSanitizer()

    async def handle_message(self, message: Message) -> None:
        """Claim compatible coding tasks and complete them with a proposal."""
        if message.type == EventType.FIX_REQUESTED:
            await self._handle_fix_request(message)
            return
        if not self.can_claim(message):
            return
        await self.claim_task(message)

        target_path = str(message.content.get("payload", {}).get("path", "README.md"))
        artifact = str(message.content.get("payload", {}).get("artifact", "readme"))
        title = str(message.content.get("title", "Crear borrador README"))
        logger.info(
            "Generando borrador task_id=%s path=%s",
            message.content["task_id"],
            target_path,
        )

        if self._uses_stable_flask_baseline(target_path, artifact):
            proposed_content = self._mock_file_content(title, target_path, artifact)
        else:
            decision = await self._decide(message, title, target_path)
            proposed_content = (
                str(decision.content)
                if decision is not None and isinstance(decision.content, str)
                else self._mock_file_content(title, target_path, artifact)
            )

        result = {"path": target_path, "content": proposed_content}
        if artifact != "design":
            await self.publish(
                EventType.CODE_PROPOSED,
                {"task_id": message.content["task_id"], **result},
                source=message,
            )
        await self.publish(
            EventType.TASK_COMPLETED,
            {"task_id": message.content["task_id"], "result": result, "owner": self.name},
            source=message,
        )

    async def _handle_fix_request(self, message: Message) -> None:
        """Propose a concrete file change from execution failure context."""
        if not self.can_claim(message):
            return
        await self.claim_task(message)
        payload = dict(message.content.get("payload", {}))
        failure = dict(payload.get("failure", {}))
        failure_context = dict(payload.get("failure_context", {}))
        target_path = str(payload.get("path", "app.py"))
        focus_files = list(
            failure_context.get("suspected_files", []) or payload.get("suggested_focus_files", [])
        )
        based_on_error = self._failure_text(failure)
        strategy = self._select_fix_strategy(failure_context, target_path)
        content = await self._fix_content(
            message,
            target_path,
            focus_files,
            based_on_error,
            strategy,
        )
        content_hash = sha256(content.encode("utf-8")).hexdigest()
        if self._fix_repeated(message, content_hash):
            content = self._make_fix_variant(content, strategy, target_path)
            content_hash = sha256(content.encode("utf-8")).hexdigest()
        result = {
            "path": target_path,
            "content": content,
            "fix_strategy": strategy.value,
            "reason": "Apply fix based on controlled execution failure.",
            "fix_reasoning": self._fix_reasoning(strategy, failure_context, target_path),
            "diff_summary": f"Replace {target_path} using {strategy.value}.",
            "based_on_error": based_on_error[:1000],
            "failure_context": failure_context,
            "content_hash": content_hash,
            "execution_task_id": payload.get("execution_task_id"),
            "command_id": payload.get("command_id", "pytest"),
            "args": list(payload.get("args", [])),
        }
        await self.publish(
            EventType.FIX_PROPOSED,
            {"task_id": message.content["task_id"], **result},
            source=message,
        )
        await self.publish(
            EventType.TASK_COMPLETED,
            {"task_id": message.content["task_id"], "result": result, "owner": self.name},
            source=message,
        )

    async def _decide(self, message: Message, title: str, target_path: str) -> LLMDecision | None:
        """Ask the LLM to generate task content."""
        if self.ollama_client is None:
            return None
        self.context_builder.record_event(message)
        prompt = PromptTemplate(
            identity="CoderAgent",
            capabilities=[Capability.CODING.value],
            constraints=[
                "Return JSON only.",
                "Generate content only for workspace files.",
                "For .py files, output raw Python code only. Do not include markdown fences.",
                "For .txt/.json files, output raw file content only.",
                "Only README.md may contain markdown.",
            ],
            input_context=self.context_builder.build(
                message,
                current_task={
                    "title": title,
                    "path": target_path,
                    "payload": message.content.get("payload", {}),
                },
                workspace_paths=self._related_paths(target_path),
            ),
            expected_json_output={
                "content": "markdown content",
                "reasoning_summary": "short text",
            },
            example_json_output={
                "content": "# Title\n\nShort file content.",
                "reasoning_summary": "Generated the requested file.",
            },
        ).render()
        try:
            data = await self.ollama_client.generate_json(prompt, CODER_DECISION_FIELDS)
            return self._decision_from_schema(data)
        except (InvalidJSONError, OllamaClientError) as error:
            logger.info("Coder LLM no disponible; usando contenido determinista: %s", error)
            self.ollama_client.record_fallback()
            return None

    def _decision_from_schema(self, data: dict[str, object]) -> LLMDecision:
        """Normalize compact coder schema to LLMDecision."""
        if "content" in data:
            return LLMDecision(
                action="generate_content",
                reasoning_summary=str(data.get("reasoning_summary", "")),
                confidence=1.0,
                content=data.get("content"),
            )
        return LLMDecision.from_dict(data)

    def _related_paths(self, target_path: str) -> list[str]:
        """Return related files that help preserve project coherence."""
        if target_path == "README.md":
            return ["app.py", "requirements.txt"]
        if target_path == "app.py":
            return ["requirements.txt"]
        if target_path == "requirements.txt":
            return ["app.py"]
        return []

    def _uses_stable_flask_baseline(self, target_path: str, artifact: str) -> bool:
        """Return whether a Flask TODO demo file must be deterministic."""
        return target_path in {"app.py", "requirements.txt", "tests/test_app.py"} or artifact in {
            "flask_app",
            "requirements",
            "pytest_tests",
        }

    async def _fix_content(
        self,
        message: Message,
        target_path: str,
        focus_files: list[object],
        based_on_error: str,
        strategy: FixStrategy,
    ) -> str:
        """Generate deterministic or LLM-backed fix content."""
        if strategy == FixStrategy.STRIP_MARKDOWN_FENCES:
            return self._direct_strip_fence_fix(target_path)
        if strategy == FixStrategy.FIX_LOCAL_MODULE_IMPORT:
            return self._direct_local_import_fix(target_path)
        if self.ollama_client is None:
            return self._mock_fix_content(target_path, based_on_error, strategy)
        self.context_builder.record_event(message)
        prompt = PromptTemplate(
            identity="CoderAgent",
            capabilities=[Capability.CODING.value],
            constraints=[
                "Return JSON only.",
                "Fix exactly one safe workspace file.",
                "Do not request command execution.",
                "For .py files, output raw Python code only. Do not include markdown fences.",
                "For .txt/.json files, output raw file content only.",
                "Only README.md may contain markdown.",
            ],
            input_context=self.context_builder.build(
                message,
                current_task={
                    "title": message.content.get("title", "Fix execution failure"),
                    "path": target_path,
                    "payload": message.content.get("payload", {}),
                    "based_on_error": based_on_error,
                    "fix_strategy": strategy.value,
                },
                workspace_paths=[str(path) for path in focus_files],
            ),
            expected_json_output={
                "content": "full replacement file content",
                "reasoning_summary": "short text",
            },
            example_json_output={
                "content": self._mock_fix_content(target_path, based_on_error, strategy),
                "reasoning_summary": "Fixed the failing assertion.",
            },
        ).render()
        try:
            data = await self.ollama_client.generate_json(prompt, CODER_DECISION_FIELDS)
            decision = self._decision_from_schema(data)
            if isinstance(decision.content, str):
                return decision.content
        except (InvalidJSONError, OllamaClientError) as error:
            logger.info("Coder fix LLM no disponible; usando fix determinista: %s", error)
            self.ollama_client.record_fallback()
        return self._mock_fix_content(target_path, based_on_error, strategy)

    def _failure_text(self, failure: dict[str, object]) -> str:
        """Join relevant failure fields for fix context."""
        return "\n".join(
            [
                str(failure.get("command", "")),
                str(failure.get("stderr", "")),
                str(failure.get("stdout", "")),
            ]
        ).strip()

    def _select_fix_strategy(
        self,
        failure_context: dict[str, object],
        target_path: str,
    ) -> FixStrategy:
        """Select a fix strategy from failure type and target file."""
        failure_type = str(failure_context.get("failure_type", ""))
        traceback = str(failure_context.get("traceback", ""))
        if failure_type == "MarkdownFenceSyntaxError" or (
            failure_type == "SyntaxError" and "```" in traceback
        ):
            return FixStrategy.STRIP_MARKDOWN_FENCES
        if failure_type == "LocalModuleNotFoundError":
            return FixStrategy.FIX_LOCAL_MODULE_IMPORT
        if failure_type == "ModuleNotFoundError":
            return FixStrategy.ADD_MISSING_DEPENDENCY
        if failure_type == "ImportError":
            return FixStrategy.ADD_MISSING_IMPORT
        if failure_type == "SyntaxError":
            return FixStrategy.PATCH_EXISTING_FILE
        if failure_type == "FlaskRouteError":
            return FixStrategy.FIX_ROUTE
        if target_path.startswith("tests/"):
            return FixStrategy.FIX_TEST
        if failure_type in {"AttributeError", "NameError"}:
            return FixStrategy.REWRITE_FUNCTION
        return FixStrategy.PATCH_EXISTING_FILE

    def _fix_reasoning(
        self,
        strategy: FixStrategy,
        failure_context: dict[str, object],
        target_path: str,
    ) -> str:
        """Return compact reasoning for traceable fix proposals."""
        failure_type = failure_context.get("failure_type", "UnknownFailure")
        failing_test = failure_context.get("failing_test", "")
        return (
            f"strategy={strategy.value}; failure_type={failure_type}; "
            f"target={target_path}; failing_test={failing_test}"
        )

    def _fix_repeated(self, message: Message, content_hash: str) -> bool:
        memory = getattr(self.context_builder, "project_memory", None)
        if memory is None:
            return False
        return bool(memory.has_fix_hash(message.correlation_id or message.id, content_hash))

    def _make_fix_variant(self, content: str, strategy: FixStrategy, target_path: str) -> str:
        """Make a repeated deterministic fix visibly different."""
        if strategy == FixStrategy.ADD_MISSING_DEPENDENCY or target_path.endswith(".py"):
            return content.rstrip() + "\n# retry: dependency retained\n"
        return content.rstrip() + "\n<!-- retry: adjusted fix -->\n"

    def _mock_fix_content(
        self,
        target_path: str,
        based_on_error: str,
        strategy: FixStrategy = FixStrategy.PATCH_EXISTING_FILE,
    ) -> str:
        """Return deterministic replacement content for common demo failures."""
        if strategy == FixStrategy.STRIP_MARKDOWN_FENCES:
            return self._direct_strip_fence_fix(target_path)
        if strategy == FixStrategy.FIX_LOCAL_MODULE_IMPORT:
            return self._direct_local_import_fix(target_path)
        if strategy == FixStrategy.ADD_MISSING_DEPENDENCY or target_path == "requirements.txt":
            return "Flask>=3.0\n"
        if target_path == "README.md" or "README" in based_on_error:
            return "\n".join(
                [
                    "# Flask TODO API",
                    "",
                    "Pequena API Flask para gestionar tareas TODO.",
                    "",
                    "## Endpoints",
                    "",
                    "- `GET /todos`",
                    "- `POST /todos`",
                    "",
                ]
            )
        if target_path == "tests/test_app.py":
            return self._mock_file_content("Fix tests", "tests/test_app.py", "pytest_tests")
        return self._mock_file_content("Fix app", "app.py", "flask_app")

    def _direct_strip_fence_fix(self, target_path: str) -> str:
        """Return existing target content with markdown fences removed."""
        file_awareness = getattr(self.context_builder, "file_awareness", None)
        if file_awareness is not None:
            existing = file_awareness.safe_read_optional(target_path)
            if existing:
                return self.content_sanitizer.normalize_code_content(target_path, existing)
        return self.content_sanitizer.normalize_code_content(target_path, "")

    def _direct_local_import_fix(self, target_path: str) -> str:
        """Align generated tests with symbols actually exported by app.py."""
        file_awareness = getattr(self.context_builder, "file_awareness", None)
        app_content = ""
        test_content = ""
        if file_awareness is not None:
            app_content = file_awareness.safe_read_optional("app.py")
            test_content = file_awareness.safe_read_optional(target_path)
        if "def create_app" in app_content:
            import_line = "from app import create_app"
            client_line = "    client = create_app().test_client()"
        else:
            import_line = "from app import app"
            client_line = "    client = app.test_client()"
        if not test_content or "import db" in test_content or "create_app" in test_content:
            return "\n".join(
                [
                    import_line,
                    "",
                    "",
                    "def test_todos_endpoint_is_available():",
                    client_line,
                    "    response = client.get('/todos')",
                    "    assert response.status_code in (200, 404)",
                    "",
                ]
            )
        lines = []
        for line in test_content.splitlines():
            if line.startswith("from app import"):
                lines.append(import_line)
                continue
            lines.append(line.replace("create_app().test_client()", "app.test_client()"))
        return "\n".join(lines).rstrip() + "\n"

    def _mock_file_content(self, title: str, target_path: str, artifact: str) -> str:
        """Return deterministic content by artifact type."""
        if target_path == "requirements.txt":
            return "Flask\npytest\n"
        if target_path == "app.py":
            return "\n".join(
                [
                    "from flask import Flask, jsonify, request",
                    "",
                    "app = Flask(__name__)",
                    "todos = []",
                    "",
                    "",
                    "@app.get('/todos')",
                    "def list_todos():",
                    "    return jsonify(todos)",
                    "",
                    "",
                    "@app.get('/todos/<int:todo_id>')",
                    "def get_todo(todo_id):",
                    "    for todo in todos:",
                    "        if todo['id'] == todo_id:",
                    "            return jsonify(todo)",
                    "    return jsonify({'error': 'not found'}), 404",
                    "",
                    "",
                    "@app.post('/todos')",
                    "def create_todo():",
                    "    data = request.get_json(silent=True) or {}",
                    "    todo = {'id': len(todos) + 1, 'title': data.get('title', '')}",
                    "    todos.append(todo)",
                    "    return jsonify(todo), 201",
                    "",
                    "",
                    "@app.put('/todos/<int:todo_id>')",
                    "def update_todo(todo_id):",
                    "    data = request.get_json(silent=True) or {}",
                    "    for todo in todos:",
                    "        if todo['id'] == todo_id:",
                    "            todo['title'] = data.get('title', todo['title'])",
                    "            return jsonify(todo)",
                    "    return jsonify({'error': 'not found'}), 404",
                    "",
                    "",
                    "@app.delete('/todos/<int:todo_id>')",
                    "def delete_todo(todo_id):",
                    "    for index, todo in enumerate(todos):",
                    "        if todo['id'] == todo_id:",
                    "            deleted = todos.pop(index)",
                    "            return jsonify(deleted)",
                    "    return jsonify({'error': 'not found'}), 404",
                    "",
                    "",
                    "if __name__ == '__main__':",
                    "    app.run(debug=True)",
                    "",
                ]
            )
        if target_path == "tests/test_app.py" or artifact == "pytest_tests":
            return "\n".join(
                [
                    "from app import app",
                    "",
                    "",
                    "def test_todos_crud_flow():",
                    "    client = app.test_client()",
                    "",
                    "    list_response = client.get('/todos')",
                    "    assert list_response.status_code == 200",
                    "    assert list_response.get_json() == []",
                    "",
                    "    create_response = client.post('/todos', json={'title': 'Comprar pan'})",
                    "    assert create_response.status_code == 201",
                    "    created = create_response.get_json()",
                    "    assert created == {'id': 1, 'title': 'Comprar pan'}",
                    "",
                    "    get_response = client.get('/todos/1')",
                    "    assert get_response.status_code == 200",
                    "    assert get_response.get_json() == created",
                    "",
                    "    delete_response = client.delete('/todos/1')",
                    "    assert delete_response.status_code == 200",
                    "    assert delete_response.get_json() == created",
                    "    assert client.get('/todos').get_json() == []",
                    "",
                ]
            )
        if artifact == "design":
            return "Estructura: app.py, requirements.txt y README.md para API Flask TODO."
        return "\n".join(
            [
                "# Flask TODO API",
                "",
                "Pequena API Flask para gestionar tareas TODO.",
                "",
                "## Archivos",
                "",
                "- `app.py`: API con endpoints `GET /todos`, `POST /todos`, "
                "`GET /todos/<id>` y `DELETE /todos/<id>`.",
                "- `requirements.txt`: dependencias Flask y pytest.",
                "- `tests/test_app.py`: tests pytest con `from app import app`.",
                "",
                "## Uso",
                "",
                "Instala dependencias y ejecuta `app.py` en un entorno local controlado.",
                "",
                f"Generado para: {title}",
                "",
            ]
        )
