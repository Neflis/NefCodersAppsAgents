"""Small in-memory store for future agent context."""

from __future__ import annotations

from typing import Any


class Memory:
    """Simple in-memory append-only store for agent context."""

    def __init__(self) -> None:
        self._items: list[Any] = []

    def add(self, item: Any) -> None:
        """Add one item."""
        self._items.append(item)

    def all(self) -> list[Any]:
        """Return a copy of all items."""
        return list(self._items)

    def clear(self) -> None:
        """Remove all items."""
        self._items.clear()
