"""Planner agent that decomposes goals into task graphs."""

from __future__ import annotations

import logging

from multi_agent_lab.agents.base_agent import BaseAgent
from multi_agent_lab.core.agent_event_logger import AgentEventLogger
from multi_agent_lab.core.capability import Capability
from multi_agent_lab.core.message import EventType, Message
from multi_agent_lab.core.message_bus import MessageBus
from multi_agent_lab.core.task_graph import Goal, TaskGraph, TaskNode
from multi_agent_lab.core.task_graph_store import TaskGraphStore
from multi_agent_lab.llm.context_builder import AgentContextBuilder
from multi_agent_lab.llm.decision import LLMDecision
from multi_agent_lab.llm.ollama_client import InvalidJSONError, OllamaClient, OllamaClientError
from multi_agent_lab.llm.prompt_template import PromptTemplate
from multi_agent_lab.llm.schemas import PLANNER_DECISION_FIELDS

logger = logging.getLogger(__name__)


class PlannerAgent(BaseAgent):
    """Agent that listens for goals and creates a dynamic task graph."""

    subscribed_events = (EventType.GOAL_SUBMITTED,)
    capabilities = (Capability.PLANNING.value,)

    def __init__(
        self,
        name: str,
        bus: MessageBus,
        graph_store: TaskGraphStore,
        event_logger: AgentEventLogger | None = None,
        llm_client: OllamaClient | None = None,
        context_builder: AgentContextBuilder | None = None,
    ) -> None:
        super().__init__(name, bus, event_logger)
        self.graph_store = graph_store
        self.llm_client = llm_client
        self.context_builder = context_builder or AgentContextBuilder(graph_store)

    async def handle_message(self, message: Message) -> None:
        """Decompose a goal into dependent tasks."""
        goal_title = str(
            message.content.get("goal", "Crear una pequena documentacion README para una app TODO")
        )
        target_path = str(message.content.get("path", "README.md"))
        allow_execution = bool(message.metadata.get("allow_execution", False))
        decision = await self._decide(message)
        graph = self._build_readme_graph(
            goal_title,
            target_path,
            message.correlation_id or message.id,
            decision,
            allow_execution,
        )
        self.graph_store.add(graph)
        logger.info("Objetivo descompuesto correlation=%s", graph.goal.correlation_id)

        await self.publish(
            EventType.GOAL_DECOMPOSED,
            {
                "goal_id": graph.goal.id,
                "goal": graph.goal.title,
                "tasks": [task.to_dict() for task in graph.nodes.values()],
            },
            source=message,
        )
        await self.publish(
            EventType.TASK_GRAPH_UPDATED,
            {"graph": graph.to_dict()},
            source=message,
        )
        for task in graph.ready_tasks():
            self.graph_store.persist(graph)
            await self.publish(
                EventType.TASK_READY,
                self._task_ready_payload(task),
                source=message,
            )

    def _build_readme_graph(
        self,
        goal_title: str,
        target_path: str,
        correlation_id: str,
        decision: LLMDecision | None = None,
        allow_execution: bool = False,
    ) -> TaskGraph:
        """Create the README demo graph."""
        graph = TaskGraph(Goal(goal_title, correlation_id))
        if self._is_flask_api_goal(goal_title):
            return self._build_flask_api_graph(graph, allow_execution)
        if self._is_python_cli_task_goal(goal_title):
            return self._build_python_cli_graph(graph, allow_execution)
        if self._is_spring_boot_goal(goal_title):
            return self._build_spring_boot_graph(graph, allow_execution)

        task_specs = (
            decision.content
            if decision is not None and isinstance(decision.content, list)
            else None
        )
        if task_specs:
            return self._build_graph_from_specs(graph, target_path, task_specs)

        draft = graph.add_task(
            TaskNode(
                title="Crear borrador README",
                description="Generar contenido Markdown inicial.",
                required_capability=Capability.CODING.value,
                payload={"path": target_path},
                priority=10,
            )
        )
        review = graph.add_task(
            TaskNode(
                title="Revisar README",
                description="Validar que el README propuesto sea aceptable.",
                required_capability=Capability.REVIEWING.value,
                payload={"path": target_path},
                dependencies={draft.id},
                priority=8,
            )
        )
        write = graph.add_task(
            TaskNode(
                title="Escribir README en workspace",
                description="Persistir el README aprobado en el workspace seguro.",
                required_capability=Capability.FILE_WRITE.value,
                payload={"path": target_path, "action": "write_file"},
                dependencies={review.id},
                priority=6,
            )
        )
        graph.add_task(
            TaskNode(
                title="Validar existencia del archivo",
                description="Comprobar de forma simulada que el archivo existe.",
                required_capability=Capability.TESTING_MOCK.value,
                payload={"path": target_path},
                dependencies={write.id},
                priority=4,
            )
        )
        return graph

    def _build_spring_boot_graph(self, graph: TaskGraph, allow_execution: bool) -> TaskGraph:
        """Create a deterministic Spring Boot minimal API graph."""
        pom = graph.add_task(
            TaskNode(
                title="Crear pom.xml",
                description="Configurar Spring Boot 3 con Java 17.",
                required_capability=Capability.CODING.value,
                payload={"path": "pom.xml", "artifact": "spring_boot_pom"},
                priority=10,
            )
        )
        app = graph.add_task(
            TaskNode(
                title="Crear DemoApplication.java",
                description="Crear clase principal Spring Boot.",
                required_capability=Capability.CODING.value,
                payload={
                    "path": "src/main/java/com/example/demo/DemoApplication.java",
                    "artifact": "spring_boot_application",
                },
                dependencies={pom.id},
                priority=9,
            )
        )
        controller = graph.add_task(
            TaskNode(
                title="Crear HealthController.java",
                description="Crear endpoint GET /health.",
                required_capability=Capability.CODING.value,
                payload={
                    "path": "src/main/java/com/example/demo/HealthController.java",
                    "artifact": "spring_boot_health_controller",
                },
                dependencies={app.id},
                priority=8,
            )
        )
        tests = graph.add_task(
            TaskNode(
                title="Crear HealthControllerTest.java",
                description="Crear test MockMvc para GET /health.",
                required_capability=Capability.CODING.value,
                payload={
                    "path": "src/test/java/com/example/demo/HealthControllerTest.java",
                    "artifact": "spring_boot_health_test",
                },
                dependencies={controller.id},
                priority=7,
            )
        )
        readme = graph.add_task(
            TaskNode(
                title="Crear README.md",
                description="Documentar API Spring Boot minima.",
                required_capability=Capability.CODING.value,
                payload={"path": "README.md", "artifact": "spring_boot_readme"},
                dependencies={tests.id},
                priority=6,
            )
        )
        review = graph.add_task(
            TaskNode(
                title="Revisar coherencia final Spring Boot",
                description="Validar pom, clases Java, test y README.",
                required_capability=Capability.REVIEWING.value,
                payload={
                    "project_review": True,
                    "paths": [
                        "pom.xml",
                        "src/main/java/com/example/demo/DemoApplication.java",
                        "src/main/java/com/example/demo/HealthController.java",
                        "src/test/java/com/example/demo/HealthControllerTest.java",
                        "README.md",
                    ],
                },
                dependencies={readme.id},
                priority=5,
            )
        )
        if allow_execution:
            graph.add_task(
                TaskNode(
                    title="Ejecutar validacion Maven",
                    description="Ejecutar mvn test dentro del workspace con whitelist estricta.",
                    required_capability=Capability.TESTING_EXECUTION.value,
                    payload={"command_id": "mvn", "args": ["test"]},
                    dependencies={review.id},
                    priority=4,
                )
            )
            return graph

        graph.add_task(
            TaskNode(
                title="Validar existencia de archivos Spring Boot",
                description="Comprobar que los archivos finales existen.",
                required_capability=Capability.TESTING_MOCK.value,
                payload={
                    "paths": [
                        "pom.xml",
                        "src/main/java/com/example/demo/DemoApplication.java",
                        "src/main/java/com/example/demo/HealthController.java",
                        "src/test/java/com/example/demo/HealthControllerTest.java",
                        "README.md",
                    ]
                },
                dependencies={review.id},
                priority=4,
            )
        )
        return graph

    def _build_python_cli_graph(self, graph: TaskGraph, allow_execution: bool) -> TaskGraph:
        """Create a deterministic Python task CLI graph."""
        requirements = graph.add_task(
            TaskNode(
                title="Crear requirements.txt",
                description="Declarar dependencias de test para la CLI Python.",
                required_capability=Capability.CODING.value,
                payload={"path": "requirements.txt", "artifact": "cli_requirements"},
                priority=10,
            )
        )
        cli = graph.add_task(
            TaskNode(
                title="Crear task_cli.py",
                description="Implementar CLI Python de tareas con argparse.",
                required_capability=Capability.CODING.value,
                payload={"path": "task_cli.py", "artifact": "python_task_cli"},
                dependencies={requirements.id},
                priority=9,
            )
        )
        tests = graph.add_task(
            TaskNode(
                title="Crear tests CLI",
                description="Crear tests pytest para add/list/done sin shell.",
                required_capability=Capability.CODING.value,
                payload={"path": "tests/test_task_cli.py", "artifact": "python_cli_tests"},
                dependencies={cli.id},
                priority=8,
            )
        )
        readme = graph.add_task(
            TaskNode(
                title="Crear README.md",
                description="Documentar uso basico de la CLI Python.",
                required_capability=Capability.CODING.value,
                payload={"path": "README.md", "artifact": "python_cli_readme"},
                dependencies={tests.id},
                priority=7,
            )
        )
        review = graph.add_task(
            TaskNode(
                title="Revisar coherencia final de CLI",
                description="Validar coherencia entre CLI, tests, requirements y README.",
                required_capability=Capability.REVIEWING.value,
                payload={
                    "project_review": True,
                    "paths": [
                        "task_cli.py",
                        "requirements.txt",
                        "tests/test_task_cli.py",
                        "README.md",
                    ],
                },
                dependencies={readme.id},
                priority=6,
            )
        )
        if allow_execution:
            graph.add_task(
                TaskNode(
                    title="Ejecutar validacion pytest",
                    description="Ejecutar pytest dentro del workspace con whitelist estricta.",
                    required_capability=Capability.TESTING_EXECUTION.value,
                    payload={"command_id": "pytest", "args": ["tests/test_task_cli.py"]},
                    dependencies={review.id},
                    priority=4,
                )
            )
            return graph

        graph.add_task(
            TaskNode(
                title="Validar existencia de archivos CLI",
                description="Comprobar que los archivos finales existen.",
                required_capability=Capability.TESTING_MOCK.value,
                payload={
                    "paths": [
                        "task_cli.py",
                        "requirements.txt",
                        "tests/test_task_cli.py",
                        "README.md",
                    ]
                },
                dependencies={review.id},
                priority=4,
            )
        )
        return graph

    def _build_flask_api_graph(self, graph: TaskGraph, allow_execution: bool) -> TaskGraph:
        """Create a multi-file Flask TODO API graph."""
        existing_files = self._workspace_files()
        expected_files = {"app.py", "requirements.txt", "README.md"}
        if expected_files.issubset(existing_files):
            return self._build_existing_flask_review_graph(graph, allow_execution)

        design = graph.add_task(
            TaskNode(
                title="Disenar estructura API Flask TODO",
                description="Definir archivos y coherencia del proyecto.",
                required_capability=Capability.CODING.value,
                payload={"path": "README.md", "artifact": "design"},
                priority=12,
            )
        )
        requirements = graph.add_task(
            TaskNode(
                title="Crear requirements.txt",
                description="Declarar dependencias de la API Flask TODO.",
                required_capability=Capability.CODING.value,
                payload={"path": "requirements.txt", "artifact": "requirements"},
                dependencies={design.id},
                priority=10,
            )
        )
        app = graph.add_task(
            TaskNode(
                title="Crear app.py",
                description="Implementar una pequena API Flask TODO sin ejecutarla.",
                required_capability=Capability.CODING.value,
                payload={"path": "app.py", "artifact": "flask_app"},
                dependencies={requirements.id},
                priority=9,
            )
        )
        readme = graph.add_task(
            TaskNode(
                title="Crear README.md",
                description="Documentar la API Flask TODO creada.",
                required_capability=Capability.CODING.value,
                payload={"path": "README.md", "artifact": "readme"},
                dependencies={app.id},
                priority=8,
            )
        )
        review = graph.add_task(
            TaskNode(
                title="Revisar coherencia final del proyecto",
                description="Validar coherencia entre app.py, requirements.txt y README.md.",
                required_capability=Capability.REVIEWING.value,
                payload={
                    "project_review": True,
                    "paths": ["app.py", "requirements.txt", "README.md"],
                },
                dependencies={readme.id},
                priority=6,
            )
        )
        if allow_execution:
            tests = graph.add_task(
                TaskNode(
                    title="Crear tests minimos",
                    description="Crear tests pytest seguros para validar archivos generados.",
                    required_capability=Capability.CODING.value,
                    payload={"path": "tests/test_app.py", "artifact": "pytest_tests"},
                    dependencies={review.id},
                    priority=5,
                )
            )
            graph.add_task(
                TaskNode(
                    title="Ejecutar validacion pytest",
                    description="Ejecutar pytest dentro del workspace con whitelist estricta.",
                    required_capability=Capability.TESTING_EXECUTION.value,
                    payload={"command_id": "pytest", "args": []},
                    dependencies={tests.id},
                    priority=4,
                )
            )
            return graph

        graph.add_task(
            TaskNode(
                title="Validar existencia de archivos",
                description="Comprobar que los archivos finales existen.",
                required_capability=Capability.TESTING_MOCK.value,
                payload={"paths": ["app.py", "requirements.txt", "README.md"]},
                dependencies={review.id},
                priority=4,
            )
        )
        return graph

    def _build_existing_flask_review_graph(
        self,
        graph: TaskGraph,
        allow_execution: bool,
    ) -> TaskGraph:
        """Create only review and validation tasks for an existing Flask project."""
        review = graph.add_task(
            TaskNode(
                title="Revisar coherencia final del proyecto existente",
                description="Validar coherencia entre archivos ya presentes.",
                required_capability=Capability.REVIEWING.value,
                payload={
                    "project_review": True,
                    "paths": ["app.py", "requirements.txt", "README.md"],
                },
                priority=6,
            )
        )
        if allow_execution:
            existing_files = self._workspace_files()
            dependency_id = review.id
            if "tests/test_app.py" not in existing_files:
                tests = graph.add_task(
                    TaskNode(
                        title="Crear tests minimos",
                        description="Crear tests pytest seguros para validar archivos existentes.",
                        required_capability=Capability.CODING.value,
                        payload={"path": "tests/test_app.py", "artifact": "pytest_tests"},
                        dependencies={review.id},
                        priority=5,
                    )
                )
                dependency_id = tests.id
            graph.add_task(
                TaskNode(
                    title="Ejecutar validacion pytest",
                    description="Ejecutar pytest dentro del workspace con whitelist estricta.",
                    required_capability=Capability.TESTING_EXECUTION.value,
                    payload={"command_id": "pytest", "args": []},
                    dependencies={dependency_id},
                    priority=4,
                )
            )
            return graph

        graph.add_task(
            TaskNode(
                title="Validar existencia de archivos existentes",
                description="Comprobar que los archivos finales existen.",
                required_capability=Capability.TESTING_MOCK.value,
                payload={"paths": ["app.py", "requirements.txt", "README.md"]},
                dependencies={review.id},
                priority=4,
            )
        )
        return graph

    def _workspace_files(self) -> set[str]:
        """Return known workspace files from context services."""
        file_awareness = getattr(self.context_builder, "file_awareness", None)
        if file_awareness is None:
            return set()
        return set(file_awareness.list_files("."))

    def _is_flask_api_goal(self, goal_title: str) -> bool:
        """Return whether a goal asks for a Flask API project."""
        lowered = goal_title.lower()
        return "flask" in lowered and "api" in lowered

    def _is_python_cli_task_goal(self, goal_title: str) -> bool:
        """Return whether a goal asks for a Python task CLI."""
        lowered = goal_title.lower()
        return "cli" in lowered and "python" in lowered and "tareas" in lowered

    def _is_spring_boot_goal(self, goal_title: str) -> bool:
        """Return whether a goal asks for a Spring Boot project."""
        return "spring boot" in goal_title.lower()

    def _build_graph_from_specs(
        self,
        graph: TaskGraph,
        target_path: str,
        task_specs: list[dict[str, object]],
    ) -> TaskGraph:
        """Create a graph from LLM task specs."""
        previous_id: str | None = None
        for index, spec in enumerate(task_specs):
            task = graph.add_task(
                TaskNode(
                    title=str(spec.get("title", f"Tarea {index + 1}")),
                    description=str(spec.get("description", "")),
                    required_capability=str(
                        spec.get("required_capability", Capability.CODING.value)
                    ),
                    payload=dict(spec.get("payload", {"path": target_path})),
                    priority=int(spec.get("priority", 10 - index)),
                )
            )
            if previous_id is not None:
                graph.add_dependency(task.id, previous_id)
            previous_id = task.id
        return graph

    async def _decide(self, message: Message) -> LLMDecision | None:
        """Ask the LLM how to decompose a goal."""
        if self.llm_client is None:
            return None
        self.context_builder.record_event(message)
        prompt = PromptTemplate(
            identity="PlannerAgent",
            capabilities=[Capability.PLANNING.value],
            constraints=["Return JSON only.", "Do not request command execution."],
            input_context=self.context_builder.build(message),
            expected_json_output={
                "tasks": [
                    {
                        "title": "task title",
                        "description": "task description",
                        "required_capability": "coding",
                        "payload": {"path": "README.md"},
                        "priority": 10,
                    }
                ],
                "reasoning_summary": "short text",
            },
            example_json_output={"tasks": [], "reasoning_summary": "Use existing graph pattern."},
        ).render()
        try:
            data = await self.llm_client.generate_json(prompt, PLANNER_DECISION_FIELDS)
            return self._decision_from_schema(data)
        except (InvalidJSONError, OllamaClientError) as error:
            logger.info("Planner LLM no disponible; usando plan determinista: %s", error)
            self.llm_client.record_fallback()
            return None

    def _decision_from_schema(self, data: dict[str, object]) -> LLMDecision:
        """Normalize compact planner schema to LLMDecision."""
        if "tasks" in data:
            return LLMDecision(
                action="decompose_goal",
                reasoning_summary=str(data.get("reasoning_summary", "")),
                confidence=1.0,
                content=data.get("tasks"),
            )
        return LLMDecision.from_dict(data)

    def _task_ready_payload(self, task: TaskNode) -> dict[str, object]:
        """Build a TASK_READY payload."""
        return {
            "task_id": task.id,
            "title": task.title,
            "required_capability": task.required_capability,
            "payload": task.payload,
            "status": task.status.value,
            "priority": task.priority,
        }
