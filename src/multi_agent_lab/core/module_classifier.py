"""Classify missing Python modules as local files or external dependencies."""

from __future__ import annotations


class ModuleClassifier:
    """Small heuristic classifier for import failures."""

    external_modules = {
        "flask",
        "sqlalchemy",
        "pytest",
        "requests",
        "pydantic",
        "fastapi",
        "jinja2",
        "werkzeug",
    }

    def is_external_dependency(self, module_name: str) -> bool:
        """Return whether a module should be fixed via requirements.txt."""
        root_module = self._root_module(module_name)
        return root_module in self.external_modules

    def is_local_module(self, module_name: str) -> bool:
        """Return whether a module is likely a workspace-local Python module."""
        root_module = self._root_module(module_name)
        return bool(root_module) and not self.is_external_dependency(root_module)

    def _root_module(self, module_name: str) -> str:
        return module_name.strip().split(".", maxsplit=1)[0].lower()
