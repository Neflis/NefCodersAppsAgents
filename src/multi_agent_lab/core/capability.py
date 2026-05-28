"""Agent capability model."""

from __future__ import annotations

from enum import StrEnum


class Capability(StrEnum):
    """Capabilities used to match ready tasks to workers."""

    PLANNING = "planning"
    CODING = "coding"
    REVIEWING = "reviewing"
    FILE_WRITE = "file_write"
    TESTING_MOCK = "testing_mock"
