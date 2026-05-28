"""Runtime launcher for safe goal execution."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal

from multi_agent_lab.agents.base_agent import BaseAgent
from multi_agent_lab.agents.coder_agent import CoderAgent
from multi_agent_lab.agents.file_agent import FileAgent
from multi_agent_lab.agents.planner_agent import PlannerAgent
from multi_agent_lab.agents.reviewer_agent import ReviewerAgent
from multi_agent_lab.agents.supervisor_agent import SupervisorAgent
from multi_agent_lab.agents.task_coordinator_agent import TaskCoordinatorAgent
from multi_agent_lab.agents.tester_agent import TesterAgent
from multi_agent_lab.agents.tester_execution_agent import TesterExecutionAgent
from multi_agent_lab.config.settings import Settings, load_settings
from multi_agent_lab.core.agent_event_logger import AgentEventLogger
from multi_agent_lab.core.event_noise import EventNoiseReducer
from multi_agent_lab.core.file_awareness import FileAwarenessService
from multi_agent_lab.core.message import EventType, Message
from multi_agent_lab.core.message_bus import MessageBus
from multi_agent_lab.core.project_memory_service import ProjectMemoryService
from multi_agent_lab.core.sqlite_store import SQLiteStore
from multi_agent_lab.core.task_graph import TaskNodeStatus
from multi_agent_lab.core.task_graph_store import TaskGraphStore
from multi_agent_lab.core.workspace_manager import WorkspaceManager
from multi_agent_lab.llm.context_builder import AgentContextBuilder
from multi_agent_lab.llm.metrics import LLMCallMetrics, LLMTraceRecorder
from multi_agent_lab.llm.ollama_client import OllamaClient
from multi_agent_lab.tools.command_tool import CommandTool
from multi_agent_lab.tools.file_tool import FileTool

RuntimeStatus = Literal["completed", "halted", "timeout"]


@dataclass(slots=True)
class RuntimeSummary:
    """Final runtime execution summary."""

    goal: str
    status: RuntimeStatus
    tasks_completed: int
    tasks_failed: int
    files_created: list[str]
    duration_seconds: float
    correlation_id: str
    terminal_event: str
    event_summary: dict[str, object] = field(default_factory=dict)
    llm_success_count: int = 0
    llm_failure_count: int = 0
    fallback_count: int = 0
    average_llm_latency: float = 0.0
    details: dict[str, object] = field(default_factory=dict)


class AgentRuntime:
    """Starts the agent network and waits for a terminal workflow event."""

    def __init__(
        self,
        goal: str,
        workspace_path: str = "workspace",
        use_mock_llm: bool = True,
        timeout_seconds: float = 10.0,
        settings: Settings | None = None,
        database_url: str | None = None,
        verbose: bool = False,
        max_context_chars: int = 6000,
        allow_execution: bool = False,
    ) -> None:
        self.goal = goal
        self.workspace_path = workspace_path
        self.use_mock_llm = use_mock_llm
        self.timeout_seconds = timeout_seconds
        self.settings = settings or load_settings()
        self.database_url = database_url or self.settings.database_url
        self.verbose = verbose
        self.max_context_chars = max_context_chars
        self.allow_execution = allow_execution

    async def run(self) -> RuntimeSummary:
        """Run one goal through the autonomous agent network."""
        started_at = perf_counter()
        store = SQLiteStore(self.database_url)
        event_logger = AgentEventLogger(store)
        project_memory = ProjectMemoryService(store)
        noise_reducer = EventNoiseReducer(verbose=self.verbose)
        bus = MessageBus(store, project_memory, noise_reducer)
        graph_store = TaskGraphStore(store)
        workspace = WorkspaceManager(self.workspace_path)
        file_tool = FileTool(workspace)
        command_tool = CommandTool(workspace)
        file_awareness = FileAwarenessService(file_tool)
        llm_metrics = LLMCallMetrics()
        trace_recorder = LLMTraceRecorder(workspace.root / ".traces")
        context_builder = AgentContextBuilder(
            graph_store,
            file_tool,
            file_awareness,
            project_memory,
            max_context_chars=self.max_context_chars,
        )
        agents = self._build_agents(
            bus,
            graph_store,
            file_tool,
            command_tool,
            file_awareness,
            event_logger,
            context_builder,
            noise_reducer,
            llm_metrics,
            trace_recorder,
        )

        terminal_inbox = await bus.subscribe_many(
            (
                EventType.TEST_EXECUTION_PASSED if self.allow_execution else EventType.TEST_PASSED,
                EventType.WORKFLOW_HALTED,
                EventType.WORKFLOW_TIMEOUT,
            )
        )
        goal_event = Message(
            sender="runtime",
            type=EventType.GOAL_SUBMITTED,
            content={"goal": self.goal, "path": "README.md"},
            metadata={
                "workspace": self.workspace_path,
                "mock": self.use_mock_llm,
                "allow_execution": self.allow_execution,
            },
        )

        try:
            for agent in agents:
                await agent.start()

            await bus.publish(
                Message(
                    sender="runtime",
                    type=EventType.WORKFLOW_STARTED,
                    content={"goal": self.goal},
                    correlation_id=goal_event.correlation_id,
                )
            )
            await bus.publish(goal_event)
            try:
                terminal_event = await asyncio.wait_for(
                    terminal_inbox.get(),
                    timeout=self.timeout_seconds,
                )
            except TimeoutError:
                terminal_event = Message(
                    sender="runtime",
                    type=EventType.WORKFLOW_TIMEOUT,
                    content={"goal": self.goal, "timeout_seconds": self.timeout_seconds},
                    correlation_id=goal_event.correlation_id,
                )
                await bus.publish(terminal_event)

            if terminal_event.type in {EventType.TEST_PASSED, EventType.TEST_EXECUTION_PASSED}:
                completed = Message(
                    sender="runtime",
                    type=EventType.WORKFLOW_COMPLETED,
                    content={"goal": self.goal},
                    correlation_id=goal_event.correlation_id,
                    causation_id=terminal_event.id,
                )
                await bus.publish(completed)
                terminal_event = completed

            return self._build_summary(
                graph_store,
                file_tool,
                goal_event.correlation_id or goal_event.id,
                terminal_event,
                perf_counter() - started_at,
                noise_reducer,
                llm_metrics,
            )
        finally:
            for agent in agents:
                await agent.stop()
            store.close()

    def _build_agents(
        self,
        bus: MessageBus,
        graph_store: TaskGraphStore,
        file_tool: FileTool,
        command_tool: CommandTool,
        file_awareness: FileAwarenessService,
        event_logger: AgentEventLogger,
        context_builder: AgentContextBuilder,
        noise_reducer: EventNoiseReducer,
        llm_metrics: LLMCallMetrics,
        trace_recorder: LLMTraceRecorder,
    ) -> list[BaseAgent]:
        """Create all agents for the runtime."""
        planner_llm = self._llm_client(
            self.settings.ollama_model_planner,
            "planner",
            llm_metrics,
            trace_recorder,
        )
        coder_llm = self._llm_client(
            self.settings.ollama_model_coder,
            "coder",
            llm_metrics,
            trace_recorder,
        )
        reviewer_llm = self._llm_client(
            self.settings.ollama_model_reviewer,
            "reviewer",
            llm_metrics,
            trace_recorder,
        )
        agents: list[BaseAgent] = [
            PlannerAgent("planner", bus, graph_store, event_logger, planner_llm, context_builder),
            CoderAgent("coder", bus, event_logger, coder_llm, context_builder),
            ReviewerAgent(
                "reviewer",
                bus,
                graph_store,
                event_logger,
                reviewer_llm,
                context_builder,
                file_awareness,
            ),
            FileAgent("file_agent", bus, file_tool, graph_store, event_logger),
            TesterAgent("tester", bus, file_tool, event_logger),
            TaskCoordinatorAgent("coordinator", bus, graph_store, event_logger),
            SupervisorAgent("supervisor", bus, event_logger, noise_reducer=noise_reducer),
        ]
        if self.allow_execution:
            agents.insert(
                -2,
                TesterExecutionAgent("tester_execution", bus, command_tool, event_logger),
            )
        return agents

    def _llm_client(
        self,
        model: str,
        agent_name: str,
        metrics: LLMCallMetrics,
        trace_recorder: LLMTraceRecorder,
    ) -> OllamaClient:
        """Create an Ollama client for one role."""
        return OllamaClient(
            self.settings.ollama_base_url,
            model,
            self.settings.ollama_timeout_seconds,
            use_mock=self.use_mock_llm,
            metrics=metrics,
            trace_recorder=trace_recorder,
            agent_name=agent_name,
        )

    def _build_summary(
        self,
        graph_store: TaskGraphStore,
        file_tool: FileTool,
        correlation_id: str,
        terminal_event: Message,
        duration_seconds: float,
        noise_reducer: EventNoiseReducer,
        llm_metrics: LLMCallMetrics,
    ) -> RuntimeSummary:
        """Build the final runtime summary."""
        completed = 0
        failed = 0
        if correlation_id in graph_store:
            graph = graph_store.get(correlation_id)
            completed = sum(
                node.status == TaskNodeStatus.COMPLETED for node in graph.nodes.values()
            )
            failed = sum(
                node.status in {TaskNodeStatus.FAILED, TaskNodeStatus.BLOCKED}
                for node in graph.nodes.values()
            )

        status: RuntimeStatus = "completed"
        if terminal_event.type == EventType.WORKFLOW_HALTED:
            status = "halted"
        if terminal_event.type == EventType.WORKFLOW_TIMEOUT:
            status = "timeout"

        return RuntimeSummary(
            goal=self.goal,
            status=status,
            tasks_completed=completed,
            tasks_failed=failed,
            files_created=file_tool.list_files("."),
            duration_seconds=duration_seconds,
            correlation_id=correlation_id,
            terminal_event=str(terminal_event.type),
            event_summary=noise_reducer.summary(),
            llm_success_count=llm_metrics.success_count,
            llm_failure_count=llm_metrics.failure_count,
            fallback_count=llm_metrics.fallback_count,
            average_llm_latency=llm_metrics.average_latency,
            details=dict(terminal_event.content or {}),
        )
