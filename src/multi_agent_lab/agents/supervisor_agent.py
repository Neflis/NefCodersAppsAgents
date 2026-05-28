"""Supervisor agent that observes all events and halts unsafe loops."""

from __future__ import annotations

import logging
from collections import Counter

from multi_agent_lab.agents.base_agent import BaseAgent
from multi_agent_lab.core.event_noise import EventNoiseReducer
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
        EventType.TEST_EXECUTION_FAILED,
        EventType.TASK_FAILED,
        EventType.TASK_RETRIED,
    }

    def __init__(
        self,
        name: str,
        bus,
        event_logger=None,
        max_events_per_correlation: int = 200,
        max_failures_per_correlation: int = 3,
        max_retries_per_correlation: int = 5,
        noise_reducer: EventNoiseReducer | None = None,
    ) -> None:
        super().__init__(name, bus, event_logger)
        self.max_events_per_correlation = max_events_per_correlation
        self.max_failures_per_correlation = max_failures_per_correlation
        self.max_retries_per_correlation = max_retries_per_correlation
        self.noise_reducer = noise_reducer
        self._event_counts: Counter[str] = Counter()
        self._failure_counts: Counter[str] = Counter()
        self._retry_counts: Counter[str] = Counter()
        self._halted: set[str] = set()
        self._fix_in_progress: set[str] = set()

    async def handle_message(self, message: Message) -> None:
        """Observe events and publish WORKFLOW_HALTED when limits are exceeded."""
        correlation_id = message.correlation_id or message.id
        if self.noise_reducer is None or self.noise_reducer.should_log(message):
            logger.info("Supervisor observo event=%s correlation=%s", message.type, correlation_id)
        if message.sender == self.name or correlation_id in self._halted:
            return

        self._event_counts[correlation_id] += 1
        if self._is_fix_progress(message):
            self._fix_in_progress.add(correlation_id)
            self._event_counts[correlation_id] = 0
        if message.type == EventType.TEST_EXECUTION_PASSED:
            self._fix_in_progress.discard(correlation_id)
        if message.type in self.failure_events:
            self._failure_counts[correlation_id] += 1
        if message.type == EventType.TASK_RETRIED:
            self._retry_counts[correlation_id] += 1

        too_many_events = self._event_counts[correlation_id] > self.max_events_per_correlation
        too_many_failures = self._failure_counts[correlation_id] > self.max_failures_per_correlation
        too_many_retries = self._retry_counts[correlation_id] > self.max_retries_per_correlation
        if too_many_events and correlation_id in self._fix_in_progress:
            return
        if too_many_events or too_many_failures or too_many_retries:
            self._halted.add(correlation_id)
            reason = "max_events_exceeded"
            if too_many_failures:
                reason = "failure_limit"
            if too_many_retries:
                reason = "retry_limit"
            await self.publish(
                EventType.WORKFLOW_HALTED,
                {
                    "correlation_id": correlation_id,
                    "reason": reason,
                },
                source=message,
            )

    def _is_fix_progress(self, message: Message) -> bool:
        """Return whether an event represents progress in the auto-fix loop."""
        return message.type in {
            EventType.FIX_REQUESTED,
            EventType.FIX_PROPOSED,
            EventType.FIX_APPLIED,
            EventType.RETEST_REQUESTED,
        }
