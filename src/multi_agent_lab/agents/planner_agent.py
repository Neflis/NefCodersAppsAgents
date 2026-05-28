"""Planner agent that decomposes goals into task graphs."""

from __future__ import annotations

import logging

from multi_agent_lab.agents.base_agent import BaseAgent
from multi_agent_lab.core.agent_event_logger import AgentEventLogger
from multi_agent_lab.core.capability import Capability
from multi_agent_lab.core.message import EventType, Message
from multi_agent_lab.core.message_bus import MessageBus
from multi_agent_lab.core.task_graph import Goal, TaskGraph, TaskNode
from multi_agent_lab.core.task_graph_store import TaskGraphStore

logger = logging.getLogger(__name__)


class PlannerAgent(BaseAgent):
    """Agent that listens for goals and creates a dynamic task graph."""

    subscribed_events = (EventType.GOAL_SUBMITTED,)
    capabilities = (Capability.PLANNING.value,)

    def __init__(
        self,
        name: str,
        bus: MessageBus,
        graph_store: TaskGraphStore,
        event_logger: AgentEventLogger | None = None,
    ) -> None:
        super().__init__(name, bus, event_logger)
        self.graph_store = graph_store

    async def handle_message(self, message: Message) -> None:
        """Decompose a goal into dependent tasks."""
        goal_title = str(
            message.content.get("goal", "Crear una pequena documentacion README para una app TODO")
        )
        target_path = str(message.content.get("path", "README.md"))
        graph = self._build_readme_graph(
            goal_title,
            target_path,
            message.correlation_id or message.id,
        )
        self.graph_store.add(graph)
        logger.info("Objetivo descompuesto correlation=%s", graph.goal.correlation_id)

        await self.publish(
            EventType.GOAL_DECOMPOSED,
            {
                "goal_id": graph.goal.id,
                "tasks": [task.to_dict() for task in graph.nodes.values()],
            },
            source=message,
        )
        await self.publish(
            EventType.TASK_GRAPH_UPDATED,
            {"graph": graph.to_dict()},
            source=message,
        )
        for task in graph.ready_tasks():
            self.graph_store.persist(graph)
            await self.publish(
                EventType.TASK_READY,
                self._task_ready_payload(task),
                source=message,
            )

    def _build_readme_graph(
        self,
        goal_title: str,
        target_path: str,
        correlation_id: str,
    ) -> TaskGraph:
        """Create the README demo graph."""
        graph = TaskGraph(Goal(goal_title, correlation_id))
        draft = graph.add_task(
            TaskNode(
                title="Crear borrador README",
                description="Generar contenido Markdown inicial.",
                required_capability=Capability.CODING.value,
                payload={"path": target_path},
                priority=10,
            )
        )
        review = graph.add_task(
            TaskNode(
                title="Revisar README",
                description="Validar que el README propuesto sea aceptable.",
                required_capability=Capability.REVIEWING.value,
                payload={"path": target_path},
                dependencies={draft.id},
                priority=8,
            )
        )
        write = graph.add_task(
            TaskNode(
                title="Escribir README en workspace",
                description="Persistir el README aprobado en el workspace seguro.",
                required_capability=Capability.FILE_WRITE.value,
                payload={"path": target_path, "action": "write_file"},
                dependencies={review.id},
                priority=6,
            )
        )
        graph.add_task(
            TaskNode(
                title="Validar existencia del archivo",
                description="Comprobar de forma simulada que el archivo existe.",
                required_capability=Capability.TESTING_MOCK.value,
                payload={"path": target_path},
                dependencies={write.id},
                priority=4,
            )
        )
        return graph

    def _task_ready_payload(self, task: TaskNode) -> dict[str, object]:
        """Build a TASK_READY payload."""
        return {
            "task_id": task.id,
            "title": task.title,
            "required_capability": task.required_capability,
            "payload": task.payload,
            "status": task.status.value,
            "priority": task.priority,
        }
