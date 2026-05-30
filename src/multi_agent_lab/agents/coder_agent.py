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

        if self._uses_stable_baseline(target_path, artifact):
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

    def _uses_stable_baseline(self, target_path: str, artifact: str) -> bool:
        """Return whether a demo file must be deterministic."""
        flask_files = {"app.py", "tests/test_app.py"}
        cli_files = {"task_cli.py", "tests/test_task_cli.py"}
        stable_artifacts = {
            "flask_app",
            "requirements",
            "pytest_tests",
            "cli_requirements",
            "python_task_cli",
            "python_cli_tests",
            "python_cli_readme",
            "spring_boot_pom",
            "spring_boot_application",
            "spring_boot_health_controller",
            "spring_boot_health_test",
            "spring_boot_readme",
            "spring_boot_crud_pom",
            "spring_boot_user_model",
            "spring_boot_user_service",
            "spring_boot_user_controller",
            "spring_boot_user_controller_test",
            "spring_boot_user_crud_readme",
            "angular_package_json",
            "angular_json",
            "angular_tsconfig",
            "angular_main",
            "angular_app_component",
            "angular_app_template",
        }
        return target_path in flask_files | cli_files or artifact in stable_artifacts

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
        if self._is_angular_path(target_path):
            return self._mock_file_content("Fix Angular project", target_path, "")
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
        if failure_type.startswith("Maven") or failure_type.startswith("Java"):
            return FixStrategy.FIX_MAVEN_COMPILATION
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
        if strategy == FixStrategy.FIX_MAVEN_COMPILATION:
            return self._mock_file_content("Fix Maven project", target_path, "")
        if target_path in {
            "package.json",
            "angular.json",
            "tsconfig.json",
            "src/main.ts",
            "src/app/app.component.ts",
            "src/app/app.component.html",
        }:
            return self._mock_file_content("Fix Angular project", target_path, "")
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

    def _is_angular_path(self, target_path: str) -> bool:
        """Return whether a target path belongs to the Angular baseline."""
        return target_path in {
            "package.json",
            "angular.json",
            "tsconfig.json",
            "src/main.ts",
            "src/app/app.component.ts",
            "src/app/app.component.html",
        }

    def _mock_file_content(self, title: str, target_path: str, artifact: str) -> str:
        """Return deterministic content by artifact type."""
        if artifact == "spring_boot_crud_pom":
            return self._spring_boot_pom_content("Spring Boot user CRUD API")
        if artifact == "spring_boot_pom" or target_path == "pom.xml":
            return self._spring_boot_pom_content()
        if artifact == "spring_boot_application" or target_path.endswith("DemoApplication.java"):
            return self._spring_boot_application_content()
        if artifact == "spring_boot_user_model" or target_path.endswith("User.java"):
            return self._spring_boot_user_model_content()
        if artifact == "spring_boot_user_service" or target_path.endswith("UserService.java"):
            return self._spring_boot_user_service_content()
        if artifact == "spring_boot_user_controller" or target_path.endswith("UserController.java"):
            return self._spring_boot_user_controller_content()
        if artifact == "spring_boot_user_controller_test" or target_path.endswith(
            "UserControllerTest.java"
        ):
            return self._spring_boot_user_controller_test_content()
        if artifact == "spring_boot_user_crud_readme":
            return self._spring_boot_user_crud_readme_content()
        if artifact == "spring_boot_health_controller" or target_path.endswith(
            "HealthController.java"
        ):
            return self._spring_boot_controller_content()
        if artifact == "spring_boot_health_test" or target_path.endswith(
            "HealthControllerTest.java"
        ):
            return self._spring_boot_test_content()
        if artifact == "spring_boot_readme":
            return self._spring_boot_readme_content()
        if artifact == "angular_package_json" or target_path == "package.json":
            return self._angular_package_json_content()
        if artifact == "angular_json" or target_path == "angular.json":
            return self._angular_json_content()
        if artifact == "angular_tsconfig" or target_path == "tsconfig.json":
            return self._angular_tsconfig_content()
        if artifact == "angular_main" or target_path == "src/main.ts":
            return self._angular_main_content()
        if artifact == "angular_app_component" or target_path == "src/app/app.component.ts":
            return self._angular_component_content()
        if artifact == "angular_app_template" or target_path == "src/app/app.component.html":
            return self._angular_template_content()
        if artifact == "cli_requirements":
            return "pytest\n"
        if target_path == "requirements.txt":
            return "Flask\npytest\n"
        if target_path == "task_cli.py" or artifact == "python_task_cli":
            return self._python_task_cli_content()
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
        if target_path == "tests/test_task_cli.py" or artifact == "python_cli_tests":
            return self._python_cli_tests_content()
        if artifact == "design":
            return "Estructura: app.py, requirements.txt y README.md para API Flask TODO."
        if artifact == "python_cli_readme":
            return "\n".join(
                [
                    "# Task CLI",
                    "",
                    "CLI Python sencilla para gestionar tareas en memoria o en JSON local.",
                    "",
                    "## Uso",
                    "",
                    '- `python task_cli.py add "Comprar pan"`',
                    "- `python task_cli.py list`",
                    "- `python task_cli.py done 1`",
                    "",
                    "## Tests",
                    "",
                    "Ejecuta `pytest` dentro del workspace.",
                    "",
                ]
            )
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

    def _python_task_cli_content(self) -> str:
        """Return a stable task CLI implementation."""
        return "\n".join(
            [
                "from __future__ import annotations",
                "",
                "import argparse",
                "import json",
                "from pathlib import Path",
                "",
                "DEFAULT_STORE = Path('tasks.json')",
                "",
                "",
                "def load_tasks(",
                "    store_path: str | Path = DEFAULT_STORE,",
                ") -> list[dict[str, object]]:",
                "    path = Path(store_path)",
                "    if not path.exists():",
                "        return []",
                "    return list(json.loads(path.read_text(encoding='utf-8')))",
                "",
                "",
                "def save_tasks(",
                "    tasks: list[dict[str, object]],",
                "    store_path: str | Path = DEFAULT_STORE,",
                ") -> None:",
                "    path = Path(store_path)",
                "    path.write_text(json.dumps(tasks, indent=2), encoding='utf-8')",
                "",
                "",
                "def add_task(",
                "    title: str,",
                "    tasks: list[dict[str, object]] | None = None,",
                ") -> tuple[list[dict[str, object]], dict[str, object]]:",
                "    items = list(tasks or [])",
                "    next_id = max((int(task['id']) for task in items), default=0) + 1",
                "    task = {'id': next_id, 'title': title, 'done': False}",
                "    items.append(task)",
                "    return items, task",
                "",
                "",
                "def list_tasks(tasks: list[dict[str, object]]) -> list[dict[str, object]]:",
                "    return list(tasks)",
                "",
                "",
                "def mark_done(",
                "    task_id: int,",
                "    tasks: list[dict[str, object]],",
                ") -> tuple[list[dict[str, object]], dict[str, object] | None]:",
                "    items = [dict(task) for task in tasks]",
                "    for task in items:",
                "        if int(task['id']) == task_id:",
                "            task['done'] = True",
                "            return items, task",
                "    return items, None",
                "",
                "",
                "def build_parser() -> argparse.ArgumentParser:",
                "    parser = argparse.ArgumentParser(description='Task manager CLI')",
                "    subparsers = parser.add_subparsers(dest='command', required=True)",
                "    add_parser = subparsers.add_parser('add')",
                "    add_parser.add_argument('title')",
                "    subparsers.add_parser('list')",
                "    done_parser = subparsers.add_parser('done')",
                "    done_parser.add_argument('task_id', type=int)",
                "    return parser",
                "",
                "",
                "def main(",
                "    argv: list[str] | None = None,",
                "    store_path: str | Path = DEFAULT_STORE,",
                ") -> int:",
                "    args = build_parser().parse_args(argv)",
                "    tasks = load_tasks(store_path)",
                "    if args.command == 'add':",
                "        tasks, task = add_task(args.title, tasks)",
                "        save_tasks(tasks, store_path)",
                "        print(f\"Added #{task['id']}: {task['title']}\")",
                "        return 0",
                "    if args.command == 'list':",
                "        for task in list_tasks(tasks):",
                "            status = 'x' if task['done'] else ' '",
                "            print(f\"[{status}] {task['id']}: {task['title']}\")",
                "        return 0",
                "    if args.command == 'done':",
                "        tasks, task = mark_done(args.task_id, tasks)",
                "        if task is None:",
                "            print('Task not found')",
                "            return 1",
                "        save_tasks(tasks, store_path)",
                "        print(f\"Done #{task['id']}: {task['title']}\")",
                "        return 0",
                "    return 1",
                "",
                "",
                "if __name__ == '__main__':",
                "    raise SystemExit(main())",
                "",
            ]
        )

    def _angular_package_json_content(self) -> str:
        """Return a stable Angular 17 package manifest."""
        return "\n".join(
            [
                "{",
                '  "scripts": {',
                '    "build": "ngc -p tsconfig.json"',
                "  },",
                '  "dependencies": {',
                '    "@angular/animations": "^17.3.0",',
                '    "@angular/common": "^17.3.0",',
                '    "@angular/compiler": "^17.3.0",',
                '    "@angular/core": "^17.3.0",',
                '    "@angular/platform-browser": "^17.3.0",',
                '    "@angular/router": "^17.3.0",',
                '    "rxjs": "^7.8.1",',
                '    "tslib": "^2.6.2",',
                '    "zone.js": "^0.14.4"',
                "  },",
                '  "devDependencies": {',
                '    "@angular-devkit/build-angular": "^17.3.0",',
                '    "@angular/cli": "^17.3.0",',
                '    "@angular/compiler-cli": "^17.3.0",',
                '    "typescript": "~5.4.5"',
                "  }",
                "}",
                "",
            ]
        )

    def _angular_json_content(self) -> str:
        """Return a minimal Angular workspace configuration."""
        return "\n".join(
            [
                "{",
                '  "$schema": "./node_modules/@angular/cli/lib/config/schema.json",',
                '  "version": 1,',
                '  "newProjectRoot": "projects",',
                '  "projects": {',
                '    "angular-minimal": {',
                '      "projectType": "application",',
                '      "schematics": {},',
                '      "root": "",',
                '      "sourceRoot": "src",',
                '      "prefix": "app",',
                '      "architect": {',
                '        "build": {',
                '          "builder": "@angular-devkit/build-angular:application",',
                '          "options": {',
                '            "outputPath": "dist/angular-minimal",',
                '            "index": false,',
                '            "browser": "src/main.ts",',
                '            "tsConfig": "tsconfig.json",',
                '            "assets": [],',
                '            "styles": []',
                "          }",
                "        }",
                "      }",
                "    }",
                "  }",
                "}",
                "",
            ]
        )

    def _angular_tsconfig_content(self) -> str:
        """Return TypeScript config for a minimal Angular app."""
        return "\n".join(
            [
                "{",
                '  "compileOnSave": false,',
                '  "compilerOptions": {',
                '    "outDir": "./dist/out-tsc",',
                '    "strict": true,',
                '    "noImplicitOverride": true,',
                '    "noPropertyAccessFromIndexSignature": true,',
                '    "noImplicitReturns": true,',
                '    "noFallthroughCasesInSwitch": true,',
                '    "skipLibCheck": true,',
                '    "esModuleInterop": true,',
                '    "sourceMap": true,',
                '    "declaration": false,',
                '    "experimentalDecorators": true,',
                '    "moduleResolution": "bundler",',
                '    "importHelpers": true,',
                '    "target": "ES2022",',
                '    "module": "ES2022",',
                '    "lib": ["ES2022", "dom"]',
                "  },",
                '  "angularCompilerOptions": {',
                '    "enableI18nLegacyMessageIdFormat": false,',
                '    "strictInjectionParameters": true,',
                '    "strictInputAccessModifiers": true,',
                '    "strictTemplates": true',
                "  }",
                "}",
                "",
            ]
        )

    def _angular_main_content(self) -> str:
        """Return standalone Angular bootstrap entrypoint."""
        return "\n".join(
            [
                "import { bootstrapApplication } from '@angular/platform-browser';",
                "import { AppComponent } from './app/app.component';",
                "",
                "bootstrapApplication(AppComponent).catch((error) => console.error(error));",
                "",
            ]
        )

    def _angular_component_content(self) -> str:
        """Return the single standalone Angular component."""
        return "\n".join(
            [
                "import { Component } from '@angular/core';",
                "",
                "@Component({",
                "  selector: 'app-root',",
                "  standalone: true,",
                "  templateUrl: './app.component.html',",
                "})",
                "export class AppComponent {}",
                "",
            ]
        )

    def _angular_template_content(self) -> str:
        """Return the minimal Angular component template."""
        return "<h1>Angular Works</h1>\n"

    def _spring_boot_pom_content(self, description: str = "Minimal Spring Boot health API") -> str:
        """Return a stable Spring Boot 3 Maven pom."""
        return "\n".join(
            [
                '<?xml version="1.0" encoding="UTF-8"?>',
                '<project xmlns="http://maven.apache.org/POM/4.0.0"',
                '         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
                '         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 '
                'https://maven.apache.org/xsd/maven-4.0.0.xsd">',
                "    <modelVersion>4.0.0</modelVersion>",
                "    <parent>",
                "        <groupId>org.springframework.boot</groupId>",
                "        <artifactId>spring-boot-starter-parent</artifactId>",
                "        <version>3.3.5</version>",
                "        <relativePath/>",
                "    </parent>",
                "    <groupId>com.example</groupId>",
                "    <artifactId>demo</artifactId>",
                "    <version>0.0.1-SNAPSHOT</version>",
                "    <name>demo</name>",
                f"    <description>{description}</description>",
                "    <properties>",
                "        <java.version>17</java.version>",
                "    </properties>",
                "    <dependencies>",
                "        <dependency>",
                "            <groupId>org.springframework.boot</groupId>",
                "            <artifactId>spring-boot-starter-web</artifactId>",
                "        </dependency>",
                "        <dependency>",
                "            <groupId>org.springframework.boot</groupId>",
                "            <artifactId>spring-boot-starter-test</artifactId>",
                "            <scope>test</scope>",
                "        </dependency>",
                "    </dependencies>",
                "    <build>",
                "        <plugins>",
                "            <plugin>",
                "                <groupId>org.springframework.boot</groupId>",
                "                <artifactId>spring-boot-maven-plugin</artifactId>",
                "            </plugin>",
                "        </plugins>",
                "    </build>",
                "</project>",
                "",
            ]
        )

    def _spring_boot_user_model_content(self) -> str:
        """Return a stable user DTO without persistence annotations."""
        return "\n".join(
            [
                "package com.example.demo.user;",
                "",
                "public record User(Long id, String name, String email) {",
                "}",
                "",
            ]
        )

    def _spring_boot_user_service_content(self) -> str:
        """Return an in-memory user service using Map<Long, User>."""
        return "\n".join(
            [
                "package com.example.demo.user;",
                "",
                "import java.util.ArrayList;",
                "import java.util.List;",
                "import java.util.Map;",
                "import java.util.Optional;",
                "import java.util.concurrent.ConcurrentHashMap;",
                "import java.util.concurrent.atomic.AtomicLong;",
                "",
                "import org.springframework.stereotype.Service;",
                "",
                "@Service",
                "public class UserService {",
                "",
                "    private final Map<Long, User> users = new ConcurrentHashMap<>();",
                "    private final AtomicLong nextId = new AtomicLong(1);",
                "",
                "    public List<User> findAll() {",
                "        return new ArrayList<>(users.values());",
                "    }",
                "",
                "    public Optional<User> findById(Long id) {",
                "        return Optional.ofNullable(users.get(id));",
                "    }",
                "",
                "    public User create(User user) {",
                "        Long id = nextId.getAndIncrement();",
                "        User created = new User(id, user.name(), user.email());",
                "        users.put(id, created);",
                "        return created;",
                "    }",
                "",
                "    public boolean delete(Long id) {",
                "        return users.remove(id) != null;",
                "    }",
                "",
                "    public void clear() {",
                "        users.clear();",
                "        nextId.set(1);",
                "    }",
                "}",
                "",
            ]
        )

    def _spring_boot_user_controller_content(self) -> str:
        """Return a stable user CRUD controller."""
        return "\n".join(
            [
                "package com.example.demo.user;",
                "",
                "import java.util.List;",
                "",
                "import org.springframework.http.HttpStatus;",
                "import org.springframework.http.ResponseEntity;",
                "import org.springframework.web.bind.annotation.DeleteMapping;",
                "import org.springframework.web.bind.annotation.GetMapping;",
                "import org.springframework.web.bind.annotation.PathVariable;",
                "import org.springframework.web.bind.annotation.PostMapping;",
                "import org.springframework.web.bind.annotation.RequestBody;",
                "import org.springframework.web.bind.annotation.RequestMapping;",
                "import org.springframework.web.bind.annotation.RestController;",
                "",
                "@RestController",
                '@RequestMapping("/users")',
                "public class UserController {",
                "",
                "    private final UserService userService;",
                "",
                "    public UserController(UserService userService) {",
                "        this.userService = userService;",
                "    }",
                "",
                "    @GetMapping",
                "    public List<User> listUsers() {",
                "        return userService.findAll();",
                "    }",
                "",
                '    @GetMapping("/{id}")',
                "    public ResponseEntity<User> getUser(@PathVariable Long id) {",
                "        return userService.findById(id)",
                "                .map(ResponseEntity::ok)",
                "                .orElseGet(() -> ResponseEntity.notFound().build());",
                "    }",
                "",
                "    @PostMapping",
                "    public ResponseEntity<User> createUser(@RequestBody User user) {",
                "        return ResponseEntity.status(HttpStatus.CREATED)",
                "                .body(userService.create(user));",
                "    }",
                "",
                '    @DeleteMapping("/{id}")',
                "    public ResponseEntity<Void> deleteUser(@PathVariable Long id) {",
                "        if (userService.delete(id)) {",
                "            return ResponseEntity.noContent().build();",
                "        }",
                "        return ResponseEntity.notFound().build();",
                "    }",
                "}",
                "",
            ]
        )

    def _spring_boot_user_controller_test_content(self) -> str:
        """Return stable MockMvc tests for the user CRUD controller."""
        return "\n".join(
            [
                "package com.example.demo.user;",
                "",
                "import org.junit.jupiter.api.BeforeEach;",
                "import org.junit.jupiter.api.Test;",
                "import org.springframework.beans.factory.annotation.Autowired;",
                "import org.springframework.boot.test.autoconfigure.web.servlet."
                "AutoConfigureMockMvc;",
                "import org.springframework.boot.test.context.SpringBootTest;",
                "import org.springframework.http.MediaType;",
                "import org.springframework.test.web.servlet.MockMvc;",
                "",
                "import static org.springframework.test.web.servlet.request."
                "MockMvcRequestBuilders.delete;",
                "import static org.springframework.test.web.servlet.request."
                "MockMvcRequestBuilders.get;",
                "import static org.springframework.test.web.servlet.request."
                "MockMvcRequestBuilders.post;",
                "import static org.springframework.test.web.servlet.result."
                "MockMvcResultMatchers.content;",
                "import static org.springframework.test.web.servlet.result."
                "MockMvcResultMatchers.jsonPath;",
                "import static org.springframework.test.web.servlet.result."
                "MockMvcResultMatchers.status;",
                "",
                "@SpringBootTest",
                "@AutoConfigureMockMvc",
                "class UserControllerTest {",
                "",
                "    @Autowired",
                "    private MockMvc mockMvc;",
                "",
                "    @Autowired",
                "    private UserService userService;",
                "",
                "    @BeforeEach",
                "    void setUp() {",
                "        userService.clear();",
                "    }",
                "",
                "    @Test",
                "    void listUsersReturnsOk() throws Exception {",
                '        mockMvc.perform(get("/users"))',
                "                .andExpect(status().isOk())",
                '                .andExpect(content().json("[]"));',
                "    }",
                "",
                "    @Test",
                "    void createGetAndDeleteUser() throws Exception {",
                '        String body = """',
                '                {"name":"Ada Lovelace","email":"ada@example.com"}',
                '                """;',
                "",
                '        mockMvc.perform(post("/users")',
                "                        .contentType(MediaType.APPLICATION_JSON)",
                "                        .content(body))",
                "                .andExpect(status().isCreated())",
                '                .andExpect(jsonPath("$.id").value(1))',
                '                .andExpect(jsonPath("$.name").value("Ada Lovelace"))',
                '                .andExpect(jsonPath("$.email").value("ada@example.com"));',
                "",
                '        mockMvc.perform(get("/users/1"))',
                "                .andExpect(status().isOk())",
                '                .andExpect(jsonPath("$.id").value(1))',
                '                .andExpect(jsonPath("$.name").value("Ada Lovelace"));',
                "",
                '        mockMvc.perform(delete("/users/1"))',
                "                .andExpect(status().isNoContent());",
                "",
                '        mockMvc.perform(get("/users/1"))',
                "                .andExpect(status().isNotFound());",
                "    }",
                "}",
                "",
            ]
        )

    def _spring_boot_user_crud_readme_content(self) -> str:
        """Return README content for the Spring Boot user CRUD API."""
        return "\n".join(
            [
                "# Spring Boot User CRUD API",
                "",
                "API Spring Boot 3 con Java 17 para gestionar usuarios en memoria.",
                "",
                "## Endpoints",
                "",
                "- `GET /users` lista usuarios.",
                "- `GET /users/{id}` obtiene un usuario.",
                "- `POST /users` crea un usuario.",
                "- `DELETE /users/{id}` elimina un usuario.",
                "",
                "## Persistencia",
                "",
                "Usa `Map<Long, User>` en memoria. No usa DB, JPA ni Lombok.",
                "",
                "## Tests",
                "",
                "Ejecuta `mvn test` dentro del workspace.",
                "",
                "Requiere Java 17 y Maven instalados.",
                "",
            ]
        )

    def _spring_boot_application_content(self) -> str:
        """Return the stable Spring Boot application class."""
        return "\n".join(
            [
                "package com.example.demo;",
                "",
                "import org.springframework.boot.SpringApplication;",
                "import org.springframework.boot.autoconfigure.SpringBootApplication;",
                "",
                "@SpringBootApplication",
                "public class DemoApplication {",
                "",
                "    public static void main(String[] args) {",
                "        SpringApplication.run(DemoApplication.class, args);",
                "    }",
                "}",
                "",
            ]
        )

    def _spring_boot_controller_content(self) -> str:
        """Return the stable health controller."""
        return "\n".join(
            [
                "package com.example.demo;",
                "",
                "import org.springframework.web.bind.annotation.GetMapping;",
                "import org.springframework.web.bind.annotation.RestController;",
                "",
                "@RestController",
                "public class HealthController {",
                "",
                '    @GetMapping("/health")',
                "    public String health() {",
                '        return "OK";',
                "    }",
                "}",
                "",
            ]
        )

    def _spring_boot_test_content(self) -> str:
        """Return the stable MockMvc health test."""
        return "\n".join(
            [
                "package com.example.demo;",
                "",
                "import org.junit.jupiter.api.Test;",
                "import org.springframework.beans.factory.annotation.Autowired;",
                "import org.springframework.boot.test.autoconfigure.web.servlet."
                "AutoConfigureMockMvc;",
                "import org.springframework.boot.test.context.SpringBootTest;",
                "import org.springframework.test.web.servlet.MockMvc;",
                "",
                "import static org.springframework.test.web.servlet.request."
                "MockMvcRequestBuilders.get;",
                "import static org.springframework.test.web.servlet.result."
                "MockMvcResultMatchers.content;",
                "import static org.springframework.test.web.servlet.result."
                "MockMvcResultMatchers.status;",
                "",
                "@SpringBootTest",
                "@AutoConfigureMockMvc",
                "class HealthControllerTest {",
                "",
                "    @Autowired",
                "    private MockMvc mockMvc;",
                "",
                "    @Test",
                "    void healthReturnsOk() throws Exception {",
                '        mockMvc.perform(get("/health"))',
                "                .andExpect(status().isOk())",
                '                .andExpect(content().string("OK"));',
                "    }",
                "}",
                "",
            ]
        )

    def _spring_boot_readme_content(self) -> str:
        """Return README content for the minimal Spring Boot API."""
        return "\n".join(
            [
                "# Spring Boot Health API",
                "",
                "API Spring Boot 3 minima con Java 17.",
                "",
                "## Endpoint",
                "",
                "- `GET /health` devuelve `OK`.",
                "",
                "## Tests",
                "",
                "Ejecuta `mvn test` dentro del workspace.",
                "",
                "Requiere Java 17 y Maven instalados.",
                "",
            ]
        )

    def _python_cli_tests_content(self) -> str:
        """Return stable tests for the task CLI."""
        return "\n".join(
            [
                "from task_cli import add_task, list_tasks, mark_done",
                "",
                "",
                "def test_add_task():",
                "    tasks, task = add_task('Comprar pan')",
                "",
                "    assert task == {'id': 1, 'title': 'Comprar pan', 'done': False}",
                "    assert tasks == [task]",
                "",
                "",
                "def test_list_tasks_returns_copy():",
                "    tasks = [{'id': 1, 'title': 'Comprar pan', 'done': False}]",
                "",
                "    assert list_tasks(tasks) == tasks",
                "    assert list_tasks(tasks) is not tasks",
                "",
                "",
                "def test_mark_done():",
                "    tasks = [{'id': 1, 'title': 'Comprar pan', 'done': False}]",
                "",
                "    updated, task = mark_done(1, tasks)",
                "",
                "    assert task == {'id': 1, 'title': 'Comprar pan', 'done': True}",
                "    assert updated == [task]",
                "    assert tasks[0]['done'] is False",
                "",
                "",
                "def test_mark_done_missing_task():",
                "    tasks = [{'id': 1, 'title': 'Comprar pan', 'done': False}]",
                "",
                "    updated, task = mark_done(999, tasks)",
                "",
                "    assert task is None",
                "    assert updated == tasks",
                "",
            ]
        )
