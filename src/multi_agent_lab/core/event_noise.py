"""Event noise reduction helpers for runtime summaries and logs."""

from __future__ import annotations

from collections import Counter

from multi_agent_lab.core.message import EventType, Message


class EventNoiseReducer:
    """Counts events and hides noisy repeats unless verbose mode is enabled."""

    noisy_events = {EventType.TASK_GRAPH_UPDATED}

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self.event_counts: Counter[str] = Counter()
        self.suppressed_counts: Counter[str] = Counter()
        self._seen_noisy: set[str] = set()

    def record(self, message: Message) -> None:
        """Record one published event."""
        self.event_counts[str(message.type)] += 1

    def should_log(self, message: Message) -> bool:
        """Return whether this event should be shown in normal logs."""
        if self.verbose:
            return True
        event_type = str(message.type)
        try:
            typed_event = EventType(str(message.type))
        except ValueError:
            typed_event = None
        if typed_event not in self.noisy_events:
            return True
        if event_type not in self._seen_noisy:
            self._seen_noisy.add(event_type)
            return True
        self.suppressed_counts[event_type] += 1
        return False

    def summary(self) -> dict[str, object]:
        """Return event count and suppression summary."""
        return {
            "event_counts": dict(sorted(self.event_counts.items())),
            "suppressed_counts": dict(sorted(self.suppressed_counts.items())),
        }
