from pathlib import Path

from multi_agent_lab.agents.reviewer_agent import ReviewerAgent
from multi_agent_lab.core.file_awareness import FileAwarenessService
from multi_agent_lab.core.workspace_manager import WorkspaceManager
from multi_agent_lab.runtime import AgentRuntime
from multi_agent_lab.tools.command_tool import CommandExecutionResult
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


async def test_spring_boot_mock_generates_minimal_baseline(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    runtime = AgentRuntime(
        "Crea una API Spring Boot minima con tests basicos",
        workspace_path=str(workspace),
        database_url=f"sqlite:///{tmp_path / 'runtime.db'}",
        use_mock_llm=True,
        timeout_seconds=10,
    )

    summary = await runtime.run()

    pom = (workspace / "pom.xml").read_text(encoding="utf-8")
    application = (
        workspace / "src" / "main" / "java" / "com" / "example" / "demo" / "DemoApplication.java"
    ).read_text(encoding="utf-8")
    controller = (
        workspace / "src" / "main" / "java" / "com" / "example" / "demo" / "HealthController.java"
    ).read_text(encoding="utf-8")
    tests = (
        workspace
        / "src"
        / "test"
        / "java"
        / "com"
        / "example"
        / "demo"
        / "HealthControllerTest.java"
    ).read_text(encoding="utf-8")
    readme = (workspace / "README.md").read_text(encoding="utf-8")
    assert summary.status == "completed"
    assert sorted(summary.files_created) == [
        "README.md",
        "pom.xml",
        "src/main/java/com/example/demo/DemoApplication.java",
        "src/main/java/com/example/demo/HealthController.java",
        "src/test/java/com/example/demo/HealthControllerTest.java",
    ]
    assert "spring-boot-starter-parent" in pom
    assert "<java.version>17</java.version>" in pom
    assert "spring-boot-starter-web" in pom
    assert "spring-boot-starter-test" in pom
    assert "spring-boot-starter-data-jpa" not in pom
    assert "lombok" not in pom.lower()
    assert "@SpringBootApplication" in application
    assert '@GetMapping("/health")' in controller
    assert 'return "OK";' in controller
    assert "MockMvc" in tests
    assert "status().isOk()" in tests
    assert 'content().string("OK")' in tests
    assert "Spring Boot" in readme
    assert "/health" in readme


async def test_spring_boot_mock_workflow_completes(tmp_path: Path) -> None:
    runtime = AgentRuntime(
        "Crea una API spring boot minima con tests basicos",
        workspace_path=str(tmp_path / "workspace"),
        database_url=f"sqlite:///{tmp_path / 'runtime.db'}",
        use_mock_llm=True,
        timeout_seconds=10,
    )

    summary = await runtime.run()

    assert summary.status == "completed"
    assert "pom.xml" in summary.files_created


async def test_spring_boot_user_crud_mock_generates_stable_baseline(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    runtime = AgentRuntime(
        "Crea una API Spring Boot CRUD de usuarios con tests basicos",
        workspace_path=str(workspace),
        database_url=f"sqlite:///{tmp_path / 'runtime.db'}",
        use_mock_llm=True,
        timeout_seconds=10,
    )

    summary = await runtime.run()

    user = (
        workspace / "src" / "main" / "java" / "com" / "example" / "demo" / "user" / "User.java"
    ).read_text(encoding="utf-8")
    service = (
        workspace
        / "src"
        / "main"
        / "java"
        / "com"
        / "example"
        / "demo"
        / "user"
        / "UserService.java"
    ).read_text(encoding="utf-8")
    controller = (
        workspace
        / "src"
        / "main"
        / "java"
        / "com"
        / "example"
        / "demo"
        / "user"
        / "UserController.java"
    ).read_text(encoding="utf-8")
    tests = (
        workspace
        / "src"
        / "test"
        / "java"
        / "com"
        / "example"
        / "demo"
        / "user"
        / "UserControllerTest.java"
    ).read_text(encoding="utf-8")
    readme = (workspace / "README.md").read_text(encoding="utf-8")
    pom = (workspace / "pom.xml").read_text(encoding="utf-8")
    assert summary.status == "completed"
    assert sorted(summary.files_created) == [
        "README.md",
        "pom.xml",
        "src/main/java/com/example/demo/DemoApplication.java",
        "src/main/java/com/example/demo/user/User.java",
        "src/main/java/com/example/demo/user/UserController.java",
        "src/main/java/com/example/demo/user/UserService.java",
        "src/test/java/com/example/demo/user/UserControllerTest.java",
    ]
    assert "spring-boot-starter-web" in pom
    assert "spring-boot-starter-data-jpa" not in pom
    assert "lombok" not in pom.lower()
    assert "record User(Long id, String name, String email)" in user
    assert "Map<Long, User>" in service
    assert "ConcurrentHashMap" in service
    assert '@RequestMapping("/users")' in controller
    assert "@GetMapping" in controller
    assert '@GetMapping("/{id}")' in controller
    assert "@PostMapping" in controller
    assert '@DeleteMapping("/{id}")' in controller
    assert "MockMvc" in tests
    assert 'get("/users")' in tests
    assert 'post("/users")' in tests
    assert 'get("/users/1")' in tests
    assert 'delete("/users/1")' in tests
    assert "status().isNoContent()" in tests
    assert "/users" in readme
    assert "Map<Long, User>" in readme


async def test_spring_boot_user_crud_mock_workflow_completes(tmp_path: Path) -> None:
    runtime = AgentRuntime(
        "Crea una API Spring Boot CRUD de usuarios con tests basicos",
        workspace_path=str(tmp_path / "workspace"),
        database_url=f"sqlite:///{tmp_path / 'runtime.db'}",
        use_mock_llm=True,
        timeout_seconds=10,
    )

    summary = await runtime.run()

    assert summary.status == "completed"
    assert "src/main/java/com/example/demo/user/UserController.java" in summary.files_created


async def test_sales_project_spec_generates_backend_baseline(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    runtime = AgentRuntime(
        "Hazme una web para registrar mis ventas de impresion 3D",
        workspace_path=str(workspace),
        database_url=f"sqlite:///{tmp_path / 'runtime.db'}",
        use_mock_llm=True,
        timeout_seconds=15,
    )

    summary = await runtime.run()

    expected_files = {
        "pom.xml",
        "README.md",
        "src/main/java/com/example/demo/DemoApplication.java",
        "src/main/java/com/example/demo/sales/Product.java",
        "src/main/java/com/example/demo/sales/Customer.java",
        "src/main/java/com/example/demo/sales/Sale.java",
        "src/main/java/com/example/demo/sales/SaleItem.java",
        "src/main/java/com/example/demo/sales/PaymentStatus.java",
        "src/main/java/com/example/demo/sales/ProductController.java",
        "src/main/java/com/example/demo/sales/CustomerController.java",
        "src/main/java/com/example/demo/sales/SaleController.java",
        "src/main/java/com/example/demo/sales/ProductService.java",
        "src/main/java/com/example/demo/sales/CustomerService.java",
        "src/main/java/com/example/demo/sales/SaleService.java",
        "src/test/java/com/example/demo/sales/SalesBackendTest.java",
    }
    assert summary.status == "completed"
    assert summary.spec_generated is True
    assert expected_files.issubset(set(summary.files_created))
    assert (workspace / ".spec" / "project_spec.json").exists()


async def test_sales_backend_contains_expected_entities_and_endpoints(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    runtime = AgentRuntime(
        "Hazme una web para registrar mis ventas de impresion 3D",
        workspace_path=str(workspace),
        database_url=f"sqlite:///{tmp_path / 'runtime.db'}",
        use_mock_llm=True,
        timeout_seconds=15,
    )

    await runtime.run()

    product = (
        workspace / "src" / "main" / "java" / "com" / "example" / "demo" / "sales" / "Product.java"
    ).read_text(encoding="utf-8")
    customer = (
        workspace / "src" / "main" / "java" / "com" / "example" / "demo" / "sales" / "Customer.java"
    ).read_text(encoding="utf-8")
    sale = (
        workspace / "src" / "main" / "java" / "com" / "example" / "demo" / "sales" / "Sale.java"
    ).read_text(encoding="utf-8")
    sale_item = (
        workspace / "src" / "main" / "java" / "com" / "example" / "demo" / "sales" / "SaleItem.java"
    ).read_text(encoding="utf-8")
    payment_status = (
        workspace
        / "src"
        / "main"
        / "java"
        / "com"
        / "example"
        / "demo"
        / "sales"
        / "PaymentStatus.java"
    ).read_text(encoding="utf-8")
    sale_controller = (
        workspace
        / "src"
        / "main"
        / "java"
        / "com"
        / "example"
        / "demo"
        / "sales"
        / "SaleController.java"
    ).read_text(encoding="utf-8")
    assert "record Product" in product
    assert "record Customer" in customer
    assert "record Sale(" in sale
    assert "record SaleItem" in sale_item
    assert "enum PaymentStatus" in payment_status
    assert '@RequestMapping("/sales")' in sale_controller
    assert '@GetMapping("/monthly-summary")' in sale_controller


async def test_sales_backend_mvn_test_runs_with_execution(tmp_path: Path, monkeypatch) -> None:
    class FakeCommandTool:
        calls: list[tuple[str, list[str]]] = []

        def __init__(self, workspace, timeout_seconds=10.0) -> None:  # noqa: ANN001
            self.workspace = workspace

        def run_pytest(self, args=None):  # noqa: ANN001
            raise AssertionError("Sales backend must use mvn test.")

        def run_command(self, command_id, args):  # noqa: ANN001
            self.calls.append((command_id, list(args)))
            return CommandExecutionResult(
                success=True,
                exit_code=0,
                stdout="mvn test ok",
                stderr="",
                duration=0.0,
            )

    monkeypatch.setattr("multi_agent_lab.runtime.CommandTool", FakeCommandTool)
    runtime = AgentRuntime(
        "Hazme una web para registrar mis ventas de impresion 3D",
        workspace_path=str(tmp_path / "workspace"),
        database_url=f"sqlite:///{tmp_path / 'runtime.db'}",
        use_mock_llm=True,
        timeout_seconds=30,
        allow_execution=True,
    )

    summary = await runtime.run()

    assert summary.status == "completed"
    assert summary.execution_success_count == 1
    assert summary.execution_failure_count == 0
    assert FakeCommandTool.calls == [("mvn", ["test"])]


async def test_angular_minimal_mock_generates_stable_baseline(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    runtime = AgentRuntime(
        "Crea una aplicación Angular mínima",
        workspace_path=str(workspace),
        database_url=f"sqlite:///{tmp_path / 'runtime.db'}",
        use_mock_llm=True,
        timeout_seconds=10,
    )

    summary = await runtime.run()

    package_json = (workspace / "package.json").read_text(encoding="utf-8")
    angular_json = (workspace / "angular.json").read_text(encoding="utf-8")
    tsconfig = (workspace / "tsconfig.json").read_text(encoding="utf-8")
    main = (workspace / "src" / "main.ts").read_text(encoding="utf-8")
    component = (workspace / "src" / "app" / "app.component.ts").read_text(encoding="utf-8")
    template = (workspace / "src" / "app" / "app.component.html").read_text(encoding="utf-8")
    assert summary.status == "completed"
    assert sorted(summary.files_created) == [
        "angular.json",
        "package.json",
        "src/app/app.component.html",
        "src/app/app.component.ts",
        "src/main.ts",
        "tsconfig.json",
    ]
    assert "@angular/core" in package_json
    assert '"build": "ngc -p tsconfig.json"' in package_json
    assert "angular-minimal" in angular_json
    assert "angularCompilerOptions" in tsconfig
    assert "bootstrapApplication(AppComponent)" in main
    assert "standalone: true" in component
    assert "Angular Works" in template


async def test_angular_minimal_mock_workflow_completes(tmp_path: Path) -> None:
    runtime = AgentRuntime(
        "Crea una aplicación Angular mínima",
        workspace_path=str(tmp_path / "workspace"),
        database_url=f"sqlite:///{tmp_path / 'runtime.db'}",
        use_mock_llm=True,
        timeout_seconds=10,
    )

    summary = await runtime.run()

    assert summary.status == "completed"
    assert "package.json" in summary.files_created


def test_reviewer_validates_spring_boot_project() -> None:
    reviewer = ReviewerAgent("reviewer", None, None)  # type: ignore[arg-type]

    feedback = reviewer._project_feedback(
        {
            "pom.xml": (
                "<project><parent><artifactId>spring-boot-starter-parent</artifactId>"
                "</parent><properties><java.version>17</java.version></properties>"
                "<dependency><artifactId>spring-boot-starter-web</artifactId></dependency>"
                "<dependency><artifactId>spring-boot-starter-test</artifactId></dependency>"
                "</project>"
            ),
            "src/main/java/com/example/demo/DemoApplication.java": (
                "@SpringBootApplication class DemoApplication {}"
            ),
            "src/main/java/com/example/demo/HealthController.java": (
                '@GetMapping("/health") String health() { return "OK"; }'
            ),
            "src/test/java/com/example/demo/HealthControllerTest.java": (
                'MockMvc mockMvc; status().isOk(); content().string("OK");'
            ),
            "README.md": "Spring Boot API con GET /health.",
        }
    )

    assert feedback == []


def test_reviewer_validates_angular_project() -> None:
    reviewer = ReviewerAgent("reviewer", None, None)  # type: ignore[arg-type]

    feedback = reviewer._project_feedback(
        {
            "package.json": (
                '{"scripts":{"build":"ngc -p tsconfig.json"},'
                '"dependencies":{"@angular/core":"^17.3.0"},'
                '"devDependencies":{"@angular/compiler-cli":"^17.3.0"}}'
            ),
            "angular.json": '{"projects":{"angular-minimal":{}}}',
            "tsconfig.json": '{"angularCompilerOptions":{}}',
            "src/main.ts": "bootstrapApplication(AppComponent);",
            "src/app/app.component.ts": "@Component({standalone: true})",
            "src/app/app.component.html": "<h1>Angular Works</h1>",
        }
    )

    assert feedback == []


def test_reviewer_validates_spring_boot_user_crud_project() -> None:
    reviewer = ReviewerAgent("reviewer", None, None)  # type: ignore[arg-type]

    feedback = reviewer._project_feedback(
        {
            "pom.xml": (
                "<project><parent><artifactId>spring-boot-starter-parent</artifactId>"
                "</parent><properties><java.version>17</java.version></properties>"
                "<dependency><artifactId>spring-boot-starter-web</artifactId></dependency>"
                "<dependency><artifactId>spring-boot-starter-test</artifactId></dependency>"
                "</project>"
            ),
            "src/main/java/com/example/demo/DemoApplication.java": (
                "@SpringBootApplication class DemoApplication {}"
            ),
            "src/main/java/com/example/demo/user/User.java": (
                "public record User(Long id, String name, String email) {}"
            ),
            "src/main/java/com/example/demo/user/UserService.java": (
                "Map<Long, User> users; findAll(); findById(1L); create(user); delete(1L);"
            ),
            "src/main/java/com/example/demo/user/UserController.java": (
                '@RequestMapping("/users") @GetMapping @GetMapping("/{id}") '
                '@PostMapping @DeleteMapping("/{id}")'
            ),
            "src/test/java/com/example/demo/user/UserControllerTest.java": (
                'MockMvc mockMvc; get("/users"); post("/users"); get("/users/1"); '
                'delete("/users/1"); status().isNoContent();'
            ),
            "README.md": "Spring Boot CRUD API con GET /users.",
        }
    )

    assert feedback == []


def test_maven_fix_keeps_spring_boot_pom() -> None:
    from multi_agent_lab.agents.coder_agent import CoderAgent
    from multi_agent_lab.core.failure_analysis import FixStrategy

    coder = CoderAgent("coder", bus=None)  # type: ignore[arg-type]

    content = coder._mock_fix_content(
        "pom.xml",
        "Could not transfer artifact spring-boot-starter-parent",
        FixStrategy.FIX_MAVEN_COMPILATION,
    )

    assert content.startswith("<?xml")
    assert "spring-boot-starter-parent" in content


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
