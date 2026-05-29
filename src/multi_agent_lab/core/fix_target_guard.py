"""Guard against applying fixes to unrelated target files."""

from __future__ import annotations

from multi_agent_lab.core.failure_analysis import FixStrategy


class FixTargetGuard:
    """Tracks and rejects clearly wrong fix targets."""

    def __init__(self) -> None:
        self.wrong_target_fix_count = 0

    def is_wrong_target(self, path: str, failure_context: object, strategy: object = "") -> bool:
        """Return whether the proposed fix target conflicts with failure context."""
        if path != "requirements.txt":
            return False
        if strategy == FixStrategy.FIX_LOCAL_MODULE_IMPORT.value:
            return True
        if not isinstance(failure_context, dict):
            return False
        return str(failure_context.get("failure_type", "")) == "LocalModuleNotFoundError"

    def record_wrong_target(self) -> None:
        """Increment wrong target counter."""
        self.wrong_target_fix_count += 1
