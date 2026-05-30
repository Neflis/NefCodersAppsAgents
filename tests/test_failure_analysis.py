from multi_agent_lab.agents.coder_agent import CoderAgent
from multi_agent_lab.core.failure_analysis import FailureAnalysisService, FixStrategy


def test_import_error_generates_add_missing_import_strategy() -> None:
    coder = CoderAgent("coder", bus=None)  # type: ignore[arg-type]

    strategy = coder._select_fix_strategy({"failure_type": "ImportError"}, "app.py")

    assert strategy == FixStrategy.ADD_MISSING_IMPORT


def test_module_not_found_generates_add_missing_dependency_strategy() -> None:
    coder = CoderAgent("coder", bus=None)  # type: ignore[arg-type]

    strategy = coder._select_fix_strategy(
        {"failure_type": "ModuleNotFoundError"}, "requirements.txt"
    )

    assert strategy == FixStrategy.ADD_MISSING_DEPENDENCY


def test_syntax_error_generates_patch_existing_file_strategy() -> None:
    coder = CoderAgent("coder", bus=None)  # type: ignore[arg-type]

    strategy = coder._select_fix_strategy({"failure_type": "SyntaxError"}, "app.py")

    assert strategy == FixStrategy.PATCH_EXISTING_FILE


def test_maven_compilation_error_generates_maven_strategy() -> None:
    coder = CoderAgent("coder", bus=None)  # type: ignore[arg-type]

    strategy = coder._select_fix_strategy(
        {"failure_type": "MavenCompilationError"},
        "src/main/java/com/example/demo/HealthController.java",
    )

    assert strategy == FixStrategy.FIX_MAVEN_COMPILATION


def test_assertion_error_reads_related_test() -> None:
    service = FailureAnalysisService()

    context = service.parse_pytest_output(
        "FAILED tests/test_app.py::test_todos - AssertionError\n"
        "tests/test_app.py:7: AssertionError",
        "",
    )

    assert context.failure_type == "AssertionError"
    assert context.failing_test == "tests/test_app.py::test_todos"
    assert "tests/test_app.py" in context.suspected_files


def test_failure_analysis_detects_traceback() -> None:
    service = FailureAnalysisService()

    context = service.parse_pytest_output(
        "",
        "Traceback (most recent call last):\n"
        '  File "app.py", line 1, in <module>\n'
        "    import missing\n"
        "ModuleNotFoundError: No module named 'missing'\n",
    )

    assert context.failure_type == "LocalModuleNotFoundError"
    assert "Traceback" in context.traceback
    assert "requirements.txt" not in context.suspected_files
    assert "missing.py" in context.suspected_files
    assert "missing" in context.suspected_symbols


def test_failure_analysis_detects_basic_maven_errors() -> None:
    service = FailureAnalysisService()

    context = service.parse_pytest_output(
        "[ERROR] COMPILATION ERROR\n"
        "[ERROR] package com.example.missing does not exist\n"
        "src/main/java/com/example/demo/HealthController.java:[3,1]",
        "",
    )

    assert context.failure_type == "MavenCompilationError"
    assert "pom.xml" in context.suspected_files
    assert "src/main/java/com/example/demo/HealthController.java" in context.suspected_files


def test_failure_analysis_detects_maven_dependency_resolution_error() -> None:
    service = FailureAnalysisService()

    context = service.parse_pytest_output(
        "[ERROR] Non-resolvable parent POM: Could not transfer artifact "
        "org.springframework.boot:spring-boot-starter-parent:pom:3.3.5",
        "",
    )

    assert context.failure_type == "MavenMissingDependency"
    assert "pom.xml" in context.suspected_files


def test_failure_analysis_detects_java_version_mismatch() -> None:
    service = FailureAnalysisService()

    context = service.parse_pytest_output(
        "[ERROR] Failed to execute goal: invalid target release: 17",
        "",
    )

    assert context.failure_type == "JavaVersionMismatch"
