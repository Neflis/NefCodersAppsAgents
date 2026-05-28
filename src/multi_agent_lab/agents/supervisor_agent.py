"""Supervisor agent that observes all events and halts unsafe loops."""

from __future__ import annotations

import logging
from collections import Counter

from multi_agent_lab.agents.base_agent import BaseAgent
from multi_agent_lab.core.message import EventType, Message
from multi_agent_lab.core.message_bus import WILDCARD_EVENT

logger = logging.getLogger(__name__)


class SupervisorAgent(BaseAgent):
    """Agent that observes the whole event network without directing it."""

    subscribed_events = (WILDCARD_EVENT,)
    failure_events = {
        EventType.FILE_WRITE_FAILED,
        EventType.REVIEW_REJECTED,
        EventType.TEST_FAILED,
        EventType.TASK_FAILED,
    }

    def __init__(
        self,
        name: str,
        bus,
        event_logger=None,
        max_events_per_correlation: int = 25,
        max_failures_per_correlation: int = 3,
    ) -> None:
        super().__init__(name, bus, event_logger)
        self.max_events_per_correlation = max_events_per_correlation
        self.max_failures_per_correlation = max_failures_per_correlation
        self._event_counts: Counter[str] = Counter()
        self._failure_counts: Counter[str] = Counter()
        self._halted: set[str] = set()

    async def handle_message(self, message: Message) -> None:
        """Observe events and publish WORKFLOW_HALTED when limits are exceeded."""
        correlation_id = message.correlation_id or message.id
        logger.info("Supervisor observo event=%s correlation=%s", message.type, correlation_id)
        if message.sender == self.name or correlation_id in self._halted:
            return

        self._event_counts[correlation_id] += 1
        if message.type in self.failure_events:
            self._failure_counts[correlation_id] += 1

        too_many_events = self._event_counts[correlation_id] > self.max_events_per_correlation
        too_many_failures = self._failure_counts[correlation_id] > self.max_failures_per_correlation
        if too_many_events or too_many_failures:
            self._halted.add(correlation_id)
            await self.publish(
                EventType.WORKFLOW_HALTED,
                {
                    "correlation_id": correlation_id,
                    "reason": "event_limit" if too_many_events else "failure_limit",
                },
                source=message,
            )
