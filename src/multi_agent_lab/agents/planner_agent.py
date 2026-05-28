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
from multi_agent_lab.llm.context_builder import AgentContextBuilder
from multi_agent_lab.llm.decision import LLMDecision
from multi_agent_lab.llm.ollama_client import InvalidJSONError, OllamaClient
from multi_agent_lab.llm.prompt_template import PromptTemplate

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
        llm_client: OllamaClient | None = None,
        context_builder: AgentContextBuilder | None = None,
    ) -> None:
        super().__init__(name, bus, event_logger)
        self.graph_store = graph_store
        self.llm_client = llm_client
        self.context_builder = context_builder or AgentContextBuilder(graph_store)

    async def handle_message(self, message: Message) -> None:
        """Decompose a goal into dependent tasks."""
        goal_title = str(
            message.content.get("goal", "Crear una pequena documentacion README para una app TODO")
        )
        target_path = str(message.content.get("path", "README.md"))
        decision = await self._decide(message)
        graph = self._build_readme_graph(
            goal_title,
            target_path,
            message.correlation_id or message.id,
            decision,
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
        decision: LLMDecision | None = None,
    ) -> TaskGraph:
        """Create the README demo graph."""
        graph = TaskGraph(Goal(goal_title, correlation_id))
        task_specs = (
            decision.content
            if decision is not None and isinstance(decision.content, list)
            else None
        )
        if task_specs:
            return self._build_graph_from_specs(graph, target_path, task_specs)

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

    def _build_graph_from_specs(
        self,
        graph: TaskGraph,
        target_path: str,
        task_specs: list[dict[str, object]],
    ) -> TaskGraph:
        """Create a graph from LLM task specs."""
        previous_id: str | None = None
        for index, spec in enumerate(task_specs):
            task = graph.add_task(
                TaskNode(
                    title=str(spec.get("title", f"Tarea {index + 1}")),
                    description=str(spec.get("description", "")),
                    required_capability=str(
                        spec.get("required_capability", Capability.CODING.value)
                    ),
                    payload=dict(spec.get("payload", {"path": target_path})),
                    priority=int(spec.get("priority", 10 - index)),
                )
            )
            if previous_id is not None:
                graph.add_dependency(task.id, previous_id)
            previous_id = task.id
        return graph

    async def _decide(self, message: Message) -> LLMDecision | None:
        """Ask the LLM how to decompose a goal."""
        if self.llm_client is None:
            return None
        self.context_builder.record_event(message)
        prompt = PromptTemplate(
            identity="PlannerAgent",
            capabilities=[Capability.PLANNING.value],
            constraints=["Return JSON only.", "Do not request command execution."],
            input_context=self.context_builder.build(message),
            expected_json_output={
                "action": "decompose_goal",
                "reasoning_summary": "short text",
                "confidence": 0.0,
                "content": [
                    {
                        "title": "task title",
                        "description": "task description",
                        "required_capability": "coding",
                        "payload": {"path": "README.md"},
                        "priority": 10,
                    }
                ],
                "events_to_publish": [],
                "task_updates": [],
            },
        ).render()
        try:
            return LLMDecision.from_dict(await self.llm_client.generate_json(prompt))
        except InvalidJSONError as error:
            logger.info("Planner LLM JSON invalido; usando plan determinista: %s", error)
            return None

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
