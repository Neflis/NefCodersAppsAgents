"""Planner agent that decomposes goals into task graphs."""

from __future__ import annotations

import logging

from multi_agent_lab.agents.base_agent import BaseAgent
from multi_agent_lab.core.agent_event_logger import AgentEventLogger
from multi_agent_lab.core.capability import Capability
from multi_agent_lab.core.message import EventType, Message
from multi_agent_lab.core.message_bus import MessageBus
from multi_agent_lab.core.project_spec import ProjectSpec
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

    subscribed_events = (EventType.GOAL_SUBMITTED, EventType.SPEC_APPROVED)
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
        if message.type == EventType.GOAL_SUBMITTED and bool(
            message.metadata.get("require_spec", False)
        ):
            return
        goal_title = str(
            message.content.get("goal", "Crear una pequena documentacion README para una app TODO")
        )
        target_path = str(message.content.get("path", "README.md"))
        allow_execution = bool(message.metadata.get("allow_execution", False))
        spec = self._project_spec_from_message(message)
        decision = await self._decide(message)
        graph = self._build_readme_graph(
            goal_title,
            target_path,
            message.correlation_id or message.id,
            decision,
            allow_execution,
            spec,
        )
        self.graph_store.add(graph)
        logger.info("Objetivo descompuesto correlation=%s", graph.goal.correlation_id)

        await self.publish(
            EventType.GOAL_DECOMPOSED,
            {
                "goal_id": graph.goal.id,
                "goal": graph.goal.title,
                "project_spec": spec.to_dict() if spec is not None else None,
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
        project_spec: ProjectSpec | None = None,
    ) -> TaskGraph:
        """Create the README demo graph."""
        graph = TaskGraph(Goal(self._goal_title(goal_title, project_spec), correlation_id))
        if self._is_sales_frontend_spec(project_spec, goal_title):
            return self._build_sales_frontend_graph(graph, allow_execution)
        if self._is_sales_backend_spec(project_spec, goal_title):
            return self._build_sales_backend_graph(graph, allow_execution)
        if self._is_flask_api_goal(goal_title):
            return self._build_flask_api_graph(graph, allow_execution)
        if self._is_python_cli_task_goal(goal_title):
            return self._build_python_cli_graph(graph, allow_execution)
        if self._is_angular_minimal_goal(goal_title):
            return self._build_angular_minimal_graph(graph, allow_execution)
        if self._is_spring_boot_crud_user_goal(goal_title):
            return self._build_spring_boot_user_crud_graph(graph, allow_execution)
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

    def _project_spec_from_message(self, message: Message) -> ProjectSpec | None:
        """Return a ProjectSpec carried by SPEC_APPROVED, if present."""
        raw_spec = message.content.get("project_spec")
        if isinstance(raw_spec, dict):
            return ProjectSpec.from_dict(raw_spec)
        return None

    def _goal_title(self, goal_title: str, project_spec: ProjectSpec | None) -> str:
        """Include spec identity in the goal title when available."""
        if project_spec is None:
            return goal_title
        return f"{project_spec.app_name}: {goal_title}"

    def _is_sales_spec(self, project_spec: ProjectSpec | None) -> bool:
        """Return whether a spec describes the 3D printing sales MVP."""
        if project_spec is None:
            return False
        required_entities = {"Product", "Customer", "Sale", "SaleItem", "Payment"}
        return required_entities.issubset(set(project_spec.entities)) and (
            "venta" in project_spec.domain_summary.lower()
            or "sales" in project_spec.app_name.lower()
        )

    def _is_sales_frontend_spec(
        self,
        project_spec: ProjectSpec | None,
        goal_title: str,
    ) -> bool:
        """Return whether a sales spec should become an Angular frontend."""
        lowered = goal_title.lower()
        return self._is_sales_spec(project_spec) and not ("backend" in lowered or "api" in lowered)

    def _is_sales_backend_spec(
        self,
        project_spec: ProjectSpec | None,
        goal_title: str,
    ) -> bool:
        """Return whether a sales spec should become a Spring Boot backend."""
        lowered = goal_title.lower()
        return self._is_sales_spec(project_spec) and ("backend" in lowered or "api" in lowered)

    def _build_sales_frontend_graph(self, graph: TaskGraph, allow_execution: bool) -> TaskGraph:
        """Create a deterministic Angular frontend for the sales ProjectSpec."""
        files = [
            ("Crear package.json", "package.json", "sales_frontend_package_json"),
            ("Crear angular.json", "angular.json", "sales_frontend_angular_json"),
            ("Crear tsconfig.json", "tsconfig.json", "sales_frontend_tsconfig"),
            ("Crear src/main.ts", "src/main.ts", "sales_frontend_main"),
            (
                "Crear app.routes.ts",
                "src/app/app.routes.ts",
                "sales_frontend_routes",
            ),
            (
                "Crear app.component.ts",
                "src/app/app.component.ts",
                "sales_frontend_app_component",
            ),
            (
                "Crear app.component.html",
                "src/app/app.component.html",
                "sales_frontend_app_template",
            ),
            (
                "Crear DashboardComponent",
                "src/app/dashboard.component.ts",
                "sales_frontend_dashboard_component",
            ),
            (
                "Crear dashboard.component.html",
                "src/app/dashboard.component.html",
                "sales_frontend_dashboard_template",
            ),
            (
                "Crear ProductsComponent",
                "src/app/products.component.ts",
                "sales_frontend_products_component",
            ),
            (
                "Crear products.component.html",
                "src/app/products.component.html",
                "sales_frontend_products_template",
            ),
            (
                "Crear CustomersComponent",
                "src/app/customers.component.ts",
                "sales_frontend_customers_component",
            ),
            (
                "Crear customers.component.html",
                "src/app/customers.component.html",
                "sales_frontend_customers_template",
            ),
            (
                "Crear SalesComponent",
                "src/app/sales.component.ts",
                "sales_frontend_sales_component",
            ),
            (
                "Crear sales.component.html",
                "src/app/sales.component.html",
                "sales_frontend_sales_template",
            ),
            (
                "Crear NewSaleComponent",
                "src/app/new-sale.component.ts",
                "sales_frontend_new_sale_component",
            ),
            (
                "Crear new-sale.component.html",
                "src/app/new-sale.component.html",
                "sales_frontend_new_sale_template",
            ),
            ("Crear README.md", "README.md", "sales_frontend_readme"),
        ]
        previous_id: str | None = None
        for index, (title, path, artifact) in enumerate(files):
            task = graph.add_task(
                TaskNode(
                    title=title,
                    description="Generar frontend Angular standalone basado en ProjectSpec.",
                    required_capability=Capability.CODING.value,
                    payload={"path": path, "artifact": artifact},
                    dependencies={previous_id} if previous_id else set(),
                    priority=30 - index,
                )
            )
            previous_id = task.id

        paths = [path for _, path, _ in files]
        review = graph.add_task(
            TaskNode(
                title="Revisar frontend ventas Angular",
                description="Validar routing, navegacion, componentes y datos mock.",
                required_capability=Capability.REVIEWING.value,
                payload={"project_review": True, "paths": paths},
                dependencies={previous_id} if previous_id else set(),
                priority=4,
            )
        )
        if allow_execution:
            install = graph.add_task(
                TaskNode(
                    title="Instalar dependencias npm",
                    description="Ejecutar npm install dentro del workspace.",
                    required_capability=Capability.TESTING_EXECUTION.value,
                    payload={"command_id": "npm", "args": ["install"]},
                    dependencies={review.id},
                    priority=3,
                )
            )
            graph.add_task(
                TaskNode(
                    title="Compilar frontend Angular",
                    description="Ejecutar npm run build dentro del workspace.",
                    required_capability=Capability.TESTING_EXECUTION.value,
                    payload={"command_id": "npm", "args": ["run", "build"]},
                    dependencies={install.id},
                    priority=2,
                )
            )
            return graph

        graph.add_task(
            TaskNode(
                title="Validar existencia frontend ventas",
                description="Comprobar que los archivos Angular del frontend existen.",
                required_capability=Capability.TESTING_MOCK.value,
                payload={"paths": paths},
                dependencies={review.id},
                priority=3,
            )
        )
        return graph

    def _build_sales_backend_graph(self, graph: TaskGraph, allow_execution: bool) -> TaskGraph:
        """Create a deterministic Spring Boot backend for the sales ProjectSpec."""
        files = [
            ("Crear pom.xml", "pom.xml", "sales_backend_pom"),
            (
                "Crear DemoApplication.java",
                "src/main/java/com/example/demo/DemoApplication.java",
                "spring_boot_application",
            ),
            (
                "Crear PaymentStatus.java",
                "src/main/java/com/example/demo/sales/PaymentStatus.java",
                "sales_backend_payment_status",
            ),
            (
                "Crear Product.java",
                "src/main/java/com/example/demo/sales/Product.java",
                "sales_backend_product",
            ),
            (
                "Crear Customer.java",
                "src/main/java/com/example/demo/sales/Customer.java",
                "sales_backend_customer",
            ),
            (
                "Crear SaleItem.java",
                "src/main/java/com/example/demo/sales/SaleItem.java",
                "sales_backend_sale_item",
            ),
            (
                "Crear Sale.java",
                "src/main/java/com/example/demo/sales/Sale.java",
                "sales_backend_sale",
            ),
            (
                "Crear ProductService.java",
                "src/main/java/com/example/demo/sales/ProductService.java",
                "sales_backend_product_service",
            ),
            (
                "Crear CustomerService.java",
                "src/main/java/com/example/demo/sales/CustomerService.java",
                "sales_backend_customer_service",
            ),
            (
                "Crear SaleService.java",
                "src/main/java/com/example/demo/sales/SaleService.java",
                "sales_backend_sale_service",
            ),
            (
                "Crear ProductController.java",
                "src/main/java/com/example/demo/sales/ProductController.java",
                "sales_backend_product_controller",
            ),
            (
                "Crear CustomerController.java",
                "src/main/java/com/example/demo/sales/CustomerController.java",
                "sales_backend_customer_controller",
            ),
            (
                "Crear SaleController.java",
                "src/main/java/com/example/demo/sales/SaleController.java",
                "sales_backend_sale_controller",
            ),
            (
                "Crear SalesBackendTest.java",
                "src/test/java/com/example/demo/sales/SalesBackendTest.java",
                "sales_backend_test",
            ),
            ("Crear README.md", "README.md", "sales_backend_readme"),
        ]
        previous_id: str | None = None
        for index, (title, path, artifact) in enumerate(files):
            task = graph.add_task(
                TaskNode(
                    title=title,
                    description="Generar backend Spring Boot MVP basado en ProjectSpec.",
                    required_capability=Capability.CODING.value,
                    payload={"path": path, "artifact": artifact},
                    dependencies={previous_id} if previous_id else set(),
                    priority=20 - index,
                )
            )
            previous_id = task.id

        paths = [path for _, path, _ in files]
        review = graph.add_task(
            TaskNode(
                title="Revisar backend ventas Spring Boot",
                description="Validar entidades, controllers, servicios, tests y README.",
                required_capability=Capability.REVIEWING.value,
                payload={"project_review": True, "paths": paths},
                dependencies={previous_id} if previous_id else set(),
                priority=4,
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
                    priority=3,
                )
            )
            return graph

        graph.add_task(
            TaskNode(
                title="Validar existencia backend ventas",
                description="Comprobar que los archivos del backend ventas existen.",
                required_capability=Capability.TESTING_MOCK.value,
                payload={"paths": paths},
                dependencies={review.id},
                priority=3,
            )
        )
        return graph

    def _build_angular_minimal_graph(self, graph: TaskGraph, allow_execution: bool) -> TaskGraph:
        """Create a deterministic Angular standalone app graph."""
        package_json = graph.add_task(
            TaskNode(
                title="Crear package.json",
                description="Configurar dependencias Angular 17 y script build.",
                required_capability=Capability.CODING.value,
                payload={"path": "package.json", "artifact": "angular_package_json"},
                priority=10,
            )
        )
        angular_json = graph.add_task(
            TaskNode(
                title="Crear angular.json",
                description="Configurar build Angular standalone.",
                required_capability=Capability.CODING.value,
                payload={"path": "angular.json", "artifact": "angular_json"},
                dependencies={package_json.id},
                priority=9,
            )
        )
        tsconfig = graph.add_task(
            TaskNode(
                title="Crear tsconfig.json",
                description="Configurar TypeScript para Angular.",
                required_capability=Capability.CODING.value,
                payload={"path": "tsconfig.json", "artifact": "angular_tsconfig"},
                dependencies={angular_json.id},
                priority=8,
            )
        )
        main = graph.add_task(
            TaskNode(
                title="Crear src/main.ts",
                description="Arrancar componente standalone.",
                required_capability=Capability.CODING.value,
                payload={"path": "src/main.ts", "artifact": "angular_main"},
                dependencies={tsconfig.id},
                priority=7,
            )
        )
        component = graph.add_task(
            TaskNode(
                title="Crear app.component.ts",
                description="Definir componente standalone unico.",
                required_capability=Capability.CODING.value,
                payload={
                    "path": "src/app/app.component.ts",
                    "artifact": "angular_app_component",
                },
                dependencies={main.id},
                priority=6,
            )
        )
        template = graph.add_task(
            TaskNode(
                title="Crear app.component.html",
                description="Mostrar Angular Works.",
                required_capability=Capability.CODING.value,
                payload={
                    "path": "src/app/app.component.html",
                    "artifact": "angular_app_template",
                },
                dependencies={component.id},
                priority=5,
            )
        )
        paths = [
            "package.json",
            "angular.json",
            "tsconfig.json",
            "src/main.ts",
            "src/app/app.component.ts",
            "src/app/app.component.html",
        ]
        review = graph.add_task(
            TaskNode(
                title="Revisar coherencia final Angular",
                description="Validar configuracion y componente standalone.",
                required_capability=Capability.REVIEWING.value,
                payload={"project_review": True, "paths": paths},
                dependencies={template.id},
                priority=4,
            )
        )
        if allow_execution:
            install = graph.add_task(
                TaskNode(
                    title="Instalar dependencias npm",
                    description="Ejecutar npm install dentro del workspace.",
                    required_capability=Capability.TESTING_EXECUTION.value,
                    payload={"command_id": "npm", "args": ["install"]},
                    dependencies={review.id},
                    priority=3,
                )
            )
            graph.add_task(
                TaskNode(
                    title="Compilar aplicacion Angular",
                    description="Ejecutar npm run build dentro del workspace.",
                    required_capability=Capability.TESTING_EXECUTION.value,
                    payload={"command_id": "npm", "args": ["run", "build"]},
                    dependencies={install.id},
                    priority=2,
                )
            )
            return graph

        graph.add_task(
            TaskNode(
                title="Validar existencia de archivos Angular",
                description="Comprobar que los archivos finales existen.",
                required_capability=Capability.TESTING_MOCK.value,
                payload={"paths": paths},
                dependencies={review.id},
                priority=2,
            )
        )
        return graph

    def _build_spring_boot_user_crud_graph(
        self, graph: TaskGraph, allow_execution: bool
    ) -> TaskGraph:
        """Create a deterministic Spring Boot in-memory user CRUD graph."""
        pom = graph.add_task(
            TaskNode(
                title="Crear pom.xml",
                description="Configurar Spring Boot 3 con Java 17.",
                required_capability=Capability.CODING.value,
                payload={"path": "pom.xml", "artifact": "spring_boot_crud_pom"},
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
        user = graph.add_task(
            TaskNode(
                title="Crear User.java",
                description="Crear modelo de usuario sin JPA.",
                required_capability=Capability.CODING.value,
                payload={
                    "path": "src/main/java/com/example/demo/user/User.java",
                    "artifact": "spring_boot_user_model",
                },
                dependencies={app.id},
                priority=8,
            )
        )
        service = graph.add_task(
            TaskNode(
                title="Crear UserService.java",
                description="Implementar almacenamiento in-memory con Map<Long, User>.",
                required_capability=Capability.CODING.value,
                payload={
                    "path": "src/main/java/com/example/demo/user/UserService.java",
                    "artifact": "spring_boot_user_service",
                },
                dependencies={user.id},
                priority=7,
            )
        )
        controller = graph.add_task(
            TaskNode(
                title="Crear UserController.java",
                description="Exponer endpoints CRUD de usuarios.",
                required_capability=Capability.CODING.value,
                payload={
                    "path": "src/main/java/com/example/demo/user/UserController.java",
                    "artifact": "spring_boot_user_controller",
                },
                dependencies={service.id},
                priority=6,
            )
        )
        tests = graph.add_task(
            TaskNode(
                title="Crear UserControllerTest.java",
                description="Crear tests MockMvc para CRUD de usuarios.",
                required_capability=Capability.CODING.value,
                payload={
                    "path": "src/test/java/com/example/demo/user/UserControllerTest.java",
                    "artifact": "spring_boot_user_controller_test",
                },
                dependencies={controller.id},
                priority=5,
            )
        )
        readme = graph.add_task(
            TaskNode(
                title="Crear README.md",
                description="Documentar API CRUD Spring Boot de usuarios.",
                required_capability=Capability.CODING.value,
                payload={"path": "README.md", "artifact": "spring_boot_user_crud_readme"},
                dependencies={tests.id},
                priority=4,
            )
        )
        review_paths = [
            "pom.xml",
            "src/main/java/com/example/demo/DemoApplication.java",
            "src/main/java/com/example/demo/user/User.java",
            "src/main/java/com/example/demo/user/UserController.java",
            "src/main/java/com/example/demo/user/UserService.java",
            "src/test/java/com/example/demo/user/UserControllerTest.java",
            "README.md",
        ]
        review = graph.add_task(
            TaskNode(
                title="Revisar coherencia final Spring Boot CRUD",
                description="Validar pom, clases Java, test y README.",
                required_capability=Capability.REVIEWING.value,
                payload={"project_review": True, "paths": review_paths},
                dependencies={readme.id},
                priority=3,
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
                    priority=2,
                )
            )
            return graph

        graph.add_task(
            TaskNode(
                title="Validar existencia de archivos Spring Boot CRUD",
                description="Comprobar que los archivos finales existen.",
                required_capability=Capability.TESTING_MOCK.value,
                payload={"paths": review_paths},
                dependencies={review.id},
                priority=2,
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

    def _is_angular_minimal_goal(self, goal_title: str) -> bool:
        """Return whether a goal asks for a minimal Angular app."""
        lowered = goal_title.lower()
        return "angular" in lowered and ("minima" in lowered or "mínima" in lowered)

    def _is_spring_boot_crud_user_goal(self, goal_title: str) -> bool:
        """Return whether a goal asks for Spring Boot user CRUD."""
        lowered = goal_title.lower()
        return (
            "spring boot" in lowered
            and "crud" in lowered
            and ("usuarios" in lowered or "users" in lowered)
        )

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
