"""Small in-memory store for future agent context."""

from __future__ import annotations

from typing import Any


class Memory:
    def __init__(self) -> None:
        self._items: list[Any] = []

    def add(self, item: Any) -> None:
        self._items.append(item)

    def all(self) -> list[Any]:
        return list(self._items)

    def clear(self) -> None:
        self._items.clear()
