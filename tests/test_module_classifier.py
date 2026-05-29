from multi_agent_lab.core.module_classifier import ModuleClassifier


def test_module_classifier_marks_local_modules() -> None:
    classifier = ModuleClassifier()

    assert classifier.is_local_module("app")
    assert classifier.is_local_module("todo")
    assert classifier.is_local_module("models")


def test_module_classifier_marks_external_dependencies() -> None:
    classifier = ModuleClassifier()

    assert classifier.is_external_dependency("flask")
    assert classifier.is_external_dependency("sqlalchemy")
    assert classifier.is_external_dependency("pytest")
