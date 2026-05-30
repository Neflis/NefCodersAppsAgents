"""Reviewer agent that validates proposed content."""

from __future__ import annotations

import logging

from multi_agent_lab.agents.base_agent import BaseAgent
from multi_agent_lab.core.capability import Capability
from multi_agent_lab.core.code_content_sanitizer import CodeContentSanitizer
from multi_agent_lab.core.file_awareness import FileAwarenessService
from multi_agent_lab.core.message import EventType, Message
from multi_agent_lab.core.task_graph_store import TaskGraphStore
from multi_agent_lab.llm.context_builder import AgentContextBuilder
from multi_agent_lab.llm.decision import LLMDecision
from multi_agent_lab.llm.ollama_client import InvalidJSONError, OllamaClient, OllamaClientError
from multi_agent_lab.llm.prompt_template import PromptTemplate
from multi_agent_lab.llm.schemas import REVIEWER_DECISION_FIELDS

logger = logging.getLogger(__name__)


class ReviewerAgent(BaseAgent):
    """Agent that claims review tasks and approves or rejects content."""

    subscribed_events = (EventType.TASK_READY,)
    capabilities = (Capability.REVIEWING.value,)

    def __init__(
        self,
        name: str,
        bus,
        graph_store: TaskGraphStore,
        event_logger=None,
        llm_client: OllamaClient | None = None,
        context_builder: AgentContextBuilder | None = None,
        file_awareness: FileAwarenessService | None = None,
        content_sanitizer: CodeContentSanitizer | None = None,
    ) -> None:
        super().__init__(name, bus, event_logger)
        self.graph_store = graph_store
        self.llm_client = llm_client
        self.context_builder = context_builder or AgentContextBuilder(graph_store)
        self.file_awareness = file_awareness
        self.content_sanitizer = content_sanitizer or CodeContentSanitizer()

    async def handle_message(self, message: Message) -> None:
        """Claim compatible review tasks and complete them."""
        if not self.can_claim(message):
            return
        await self.claim_task(message)

        if message.content.get("payload", {}).get("project_review"):
            await self._review_project(message)
            return

        graph = self.graph_store.get(message.correlation_id or "")
        dependency_results = graph.dependency_results(str(message.content["task_id"]))
        draft = dependency_results[-1] if dependency_results else {}
        content = str(draft.get("content", ""))
        path = str(
            draft.get(
                "path",
                message.content.get("payload", {}).get("path", "README.md"),
            )
        )
        logger.info("Revisando contenido task_id=%s path=%s", message.content["task_id"], path)

        decision = await self._decide(message, content, path)
        approved = self._is_valid_content(path, content)
        rejection_reason = "Contenido invalido"
        if not self._has_valid_code_format(path, content):
            approved = False
            rejection_reason = "Python files must not include markdown code fences."
        if decision is not None:
            approved = decision.action == "approve" and self._has_valid_code_format(
                path,
                content,
            )

        if not approved:
            await self.publish(
                EventType.REVIEW_REJECTED,
                {
                    "task_id": message.content["task_id"],
                    "path": path,
                    "reason": rejection_reason,
                },
                source=message,
            )
            await self.publish(
                EventType.TASK_FAILED,
                {"task_id": message.content["task_id"], "error": rejection_reason},
                source=message,
            )
            return

        result = {"path": path, "content": content, "approved": True}
        await self.publish(
            EventType.REVIEW_APPROVED,
            {"task_id": message.content["task_id"], **result},
            source=message,
        )
        await self.publish(
            EventType.TASK_COMPLETED,
            {"task_id": message.content["task_id"], "result": result, "owner": self.name},
            source=message,
        )

    async def _review_project(self, message: Message) -> None:
        """Review final multi-file project coherence."""
        paths = list(message.content.get("payload", {}).get("paths", []))
        files = self.file_awareness.read_relevant_files(paths) if self.file_awareness else {}
        feedback = self._project_feedback(files)
        if feedback:
            await self.publish(
                EventType.PROJECT_REVIEW_REJECTED,
                {"task_id": message.content["task_id"], "feedback": feedback},
                source=message,
            )
            await self.publish(
                EventType.TASK_FAILED,
                {"task_id": message.content["task_id"], "error": "; ".join(feedback)},
                source=message,
            )
            return

        result = {"paths": paths, "approved": True}
        await self.publish(
            EventType.PROJECT_REVIEW_APPROVED,
            {"task_id": message.content["task_id"], **result},
            source=message,
        )
        await self.publish(
            EventType.TASK_COMPLETED,
            {"task_id": message.content["task_id"], "result": result, "owner": self.name},
            source=message,
        )

    def _project_feedback(self, files: dict[str, str]) -> list[str]:
        """Return actionable feedback for a generated multi-file project."""
        if "src/main/java/com/example/demo/user/UserController.java" in files:
            return self._spring_boot_user_crud_feedback(files)
        if "pom.xml" in files:
            return self._spring_boot_feedback(files)
        if "task_cli.py" in files:
            return self._python_cli_feedback(files)
        return self._flask_project_feedback(files)

    def _spring_boot_user_crud_feedback(self, files: dict[str, str]) -> list[str]:
        """Return actionable feedback for a Spring Boot user CRUD project."""
        feedback = self._spring_boot_common_feedback(files)
        user = files.get("src/main/java/com/example/demo/user/User.java", "")
        service = files.get("src/main/java/com/example/demo/user/UserService.java", "")
        controller = files.get("src/main/java/com/example/demo/user/UserController.java", "")
        tests = files.get("src/test/java/com/example/demo/user/UserControllerTest.java", "")
        readme = files.get("README.md", "")
        if "record User" not in user and "class User" not in user:
            feedback.append("User.java debe definir el modelo User.")
        if "Map<Long, User>" not in service:
            feedback.append("UserService.java debe usar Map<Long, User> en memoria.")
        for token in ("findAll", "findById", "create", "delete"):
            if token not in service:
                feedback.append(f"UserService.java debe implementar {token}.")
        for route in (
            '@RequestMapping("/users")',
            "@GetMapping",
            '@GetMapping("/{id}")',
            "@PostMapping",
            '@DeleteMapping("/{id}")',
        ):
            if route not in controller:
                feedback.append(f"UserController.java debe exponer {route}.")
        for expectation in (
            'get("/users")',
            'post("/users")',
            'get("/users/1")',
            'delete("/users/1")',
            "status().isNoContent()",
        ):
            if expectation not in tests:
                feedback.append("UserControllerTest.java debe probar el CRUD con MockMvc.")
                break
        if "crud" not in readme.lower() or "/users" not in readme:
            feedback.append("README.md debe documentar la API CRUD de usuarios.")
        return feedback

    def _spring_boot_feedback(self, files: dict[str, str]) -> list[str]:
        """Return actionable feedback for a minimal Spring Boot project."""
        feedback = self._spring_boot_common_feedback(files)
        application = files.get("src/main/java/com/example/demo/DemoApplication.java", "")
        controller = files.get("src/main/java/com/example/demo/HealthController.java", "")
        tests = files.get("src/test/java/com/example/demo/HealthControllerTest.java", "")
        readme = files.get("README.md", "")
        if "@SpringBootApplication" not in application:
            feedback.append("DemoApplication.java debe definir la aplicacion Spring Boot.")
        if '@GetMapping("/health")' not in controller or 'return "OK";' not in controller:
            feedback.append("HealthController.java debe exponer GET /health con respuesta OK.")
        if "MockMvc" not in tests:
            feedback.append("HealthControllerTest.java debe usar MockMvc.")
        if "status().isOk()" not in tests or 'content().string("OK")' not in tests:
            feedback.append("HealthControllerTest.java debe validar 200 y body OK.")
        if "spring boot" not in readme.lower() or "/health" not in readme:
            feedback.append("README.md debe documentar Spring Boot y GET /health.")
        return feedback

    def _spring_boot_common_feedback(self, files: dict[str, str]) -> list[str]:
        """Return common feedback for Spring Boot Maven projects."""
        feedback: list[str] = []
        pom = files.get("pom.xml", "")
        application = files.get("src/main/java/com/example/demo/DemoApplication.java", "")
        if "spring-boot-starter-parent" not in pom:
            feedback.append("pom.xml debe usar Spring Boot 3.")
        if "<java.version>17</java.version>" not in pom:
            feedback.append("pom.xml debe configurar Java 17.")
        if "spring-boot-starter-web" not in pom:
            feedback.append("pom.xml debe incluir spring-boot-starter-web.")
        if "spring-boot-starter-test" not in pom:
            feedback.append("pom.xml debe incluir spring-boot-starter-test.")
        forbidden = ("spring-boot-starter-data-jpa", "lombok", "postgresql", "mysql")
        if any(token in pom.lower() for token in forbidden):
            feedback.append("pom.xml no debe incluir DB, JPA ni Lombok.")
        if "@SpringBootApplication" not in application:
            feedback.append("DemoApplication.java debe definir la aplicacion Spring Boot.")
        return feedback

    def _flask_project_feedback(self, files: dict[str, str]) -> list[str]:
        """Return actionable feedback for a multi-file Flask project."""
        feedback: list[str] = []
        app = files.get("app.py", "")
        requirements = files.get("requirements.txt", "")
        readme = files.get("README.md", "")
        if "from flask import" in app.lower() and "flask" not in requirements.lower():
            feedback.append("requirements.txt debe incluir Flask.")
        if "flask" not in readme.lower() or "todo" not in readme.lower():
            feedback.append("README.md debe describir la API Flask TODO.")
        if "GET /todos" not in readme and "/todos" not in readme:
            feedback.append("README.md debe mencionar los endpoints TODO.")
        return feedback

    def _python_cli_feedback(self, files: dict[str, str]) -> list[str]:
        """Return actionable feedback for a Python task CLI project."""
        feedback: list[str] = []
        cli = files.get("task_cli.py", "")
        tests = files.get("tests/test_task_cli.py", "")
        requirements = files.get("requirements.txt", "")
        readme = files.get("README.md", "")
        if "argparse" not in cli:
            feedback.append("task_cli.py debe usar argparse.")
        for name in ("add_task", "list_tasks", "mark_done"):
            if f"def {name}" not in cli:
                feedback.append(f"task_cli.py debe definir {name}.")
        if "pytest" not in requirements.lower():
            feedback.append("requirements.txt debe incluir pytest.")
        if "from task_cli import" not in tests:
            feedback.append("tests/test_task_cli.py debe importar funciones de task_cli.")
        forbidden = ("import db", "Todo", "create_app")
        if any(token in tests for token in forbidden):
            feedback.append("tests/test_task_cli.py no debe importar db, Todo ni create_app.")
        if "cli" not in readme.lower() or "add" not in readme.lower():
            feedback.append("README.md debe explicar uso basico de la CLI.")
        return feedback

    def _is_valid_content(self, path: str, content: str) -> bool:
        """Validate generated content before approving file writing."""
        return (
            bool(content.strip())
            and len(content.encode("utf-8")) <= 1024 * 1024
            and self._has_valid_code_format(path, content)
        )

    def _has_valid_code_format(self, path: str, content: str) -> bool:
        """Return whether code files avoid markdown code fences."""
        try:
            self.content_sanitizer.validate_no_markdown_fences(path, content)
        except ValueError:
            return False
        return True

    async def _decide(self, message: Message, content: str, path: str) -> LLMDecision | None:
        """Ask the LLM to approve or reject content."""
        if self.llm_client is None:
            return None
        self.context_builder.record_event(message)
        prompt = PromptTemplate(
            identity="ReviewerAgent",
            capabilities=[Capability.REVIEWING.value],
            constraints=["Return JSON only.", "Approve only safe non-empty content."],
            input_context=self.context_builder.build(
                message,
                current_task={"path": path, "content": content},
            ),
            expected_json_output={
                "approved": True,
                "feedback": "",
                "reasoning_summary": "short text",
            },
            example_json_output={
                "approved": True,
                "feedback": "",
                "reasoning_summary": "Safe non-empty content.",
            },
        ).render()
        try:
            data = await self.llm_client.generate_json(prompt, REVIEWER_DECISION_FIELDS)
            return self._decision_from_schema(data)
        except (InvalidJSONError, OllamaClientError) as error:
            logger.info("Reviewer LLM no disponible; usando validacion determinista: %s", error)
            self.llm_client.record_fallback()
            return None

    def _decision_from_schema(self, data: dict[str, object]) -> LLMDecision:
        """Normalize compact reviewer schema to LLMDecision."""
        if "approved" in data:
            approved = bool(data.get("approved"))
            return LLMDecision(
                action="approve" if approved else "reject",
                reasoning_summary=str(data.get("reasoning_summary", "")),
                confidence=1.0,
                content={"approved": approved, "feedback": str(data.get("feedback", ""))},
            )
        return LLMDecision.from_dict(data)
