"""Agent that converts vague goals into structured project specs."""

from __future__ import annotations

import logging

from multi_agent_lab.agents.base_agent import BaseAgent
from multi_agent_lab.core.agent_event_logger import AgentEventLogger
from multi_agent_lab.core.message import EventType, Message
from multi_agent_lab.core.message_bus import MessageBus
from multi_agent_lab.core.project_spec import ProjectSpec, generic_mvp_spec, sales_3d_printing_spec

logger = logging.getLogger(__name__)

SPEC_PATH = ".spec/project_spec.json"


class SpecAgent(BaseAgent):
    """Agent that prepares an MVP functional specification before planning."""

    subscribed_events = (EventType.GOAL_SUBMITTED,)

    def __init__(
        self,
        name: str,
        bus: MessageBus,
        event_logger: AgentEventLogger | None = None,
    ) -> None:
        super().__init__(name, bus, event_logger)

    async def handle_message(self, message: Message) -> None:
        """Generate and approve a structured project spec for the goal."""
        if not bool(message.metadata.get("require_spec", False)):
            return
        goal = str(message.content.get("goal", ""))
        await self.publish(
            EventType.SPEC_REQUESTED,
            {"goal": goal},
            source=message,
        )
        spec = self._spec_from_goal(goal)
        spec_payload = spec.to_dict()
        logger.info("Spec generado app_name=%s", spec.app_name)
        await self.publish(
            EventType.CODE_PROPOSED,
            {
                "task_id": "project-spec",
                "path": SPEC_PATH,
                "content": spec.to_json(),
            },
            source=message,
        )
        await self.publish(
            EventType.SPEC_GENERATED,
            {"goal": goal, "path": SPEC_PATH, "project_spec": spec_payload},
            source=message,
        )
        await self.publish(
            EventType.SPEC_APPROVED,
            {
                "goal": goal,
                "path": message.content.get("path", "README.md"),
                "spec_path": SPEC_PATH,
                "project_spec": spec_payload,
            },
            source=message,
            metadata={
                "allow_execution": bool(message.metadata.get("allow_execution", False)),
                "mock": bool(message.metadata.get("mock", True)),
            },
        )

    def _spec_from_goal(self, goal: str) -> ProjectSpec:
        """Select a deterministic MVP spec for a user goal."""
        lowered = goal.lower()
        if "venta" in lowered and (
            "impresion 3d" in lowered or "impresión 3d" in lowered or "3d" in lowered
        ):
            return sales_3d_printing_spec()
        return generic_mvp_spec(goal)
