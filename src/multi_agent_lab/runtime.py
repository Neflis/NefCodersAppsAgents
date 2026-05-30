"""Runtime launcher for safe goal execution."""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal

from multi_agent_lab.agents.base_agent import BaseAgent
from multi_agent_lab.agents.coder_agent import CoderAgent
from multi_agent_lab.agents.dependency_installer_agent import DependencyInstallerAgent
from multi_agent_lab.agents.file_agent import FileAgent
from multi_agent_lab.agents.planner_agent import PlannerAgent
from multi_agent_lab.agents.reviewer_agent import ReviewerAgent
from multi_agent_lab.agents.supervisor_agent import SupervisorAgent
from multi_agent_lab.agents.task_coordinator_agent import TaskCoordinatorAgent
from multi_agent_lab.agents.tester_agent import TesterAgent
from multi_agent_lab.agents.tester_execution_agent import TesterExecutionAgent
from multi_agent_lab.config.settings import Settings, load_settings
from multi_agent_lab.core.agent_event_logger import AgentEventLogger
from multi_agent_lab.core.code_content_sanitizer import CodeContentSanitizer
from multi_agent_lab.core.event_noise import EventNoiseReducer
from multi_agent_lab.core.file_awareness import FileAwarenessService
from multi_agent_lab.core.file_path_normalizer import FilePathNormalizer
from multi_agent_lab.core.fix_target_guard import FixTargetGuard
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
    execution_success_count: int = 0
    execution_failure_count: int = 0
    fix_attempts: int = 0
    final_failure_reason: str = ""
    detected_failure_types: list[str] = field(default_factory=list)
    fixes_attempted: int = 0
    repeated_failures: list[str] = field(default_factory=list)
    invalid_paths_detected: int = 0
    invalid_paths_ignored: int = 0
    sanitized_files_count: int = 0
    wrong_target_fix_count: int = 0
    installed_dependencies: int = 0
    dependency_install_failures: int = 0
    patches_applied: int = 0
    patches_failed: int = 0
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
        max_events: int | None = None,
        max_fix_attempts: int | None = None,
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
        self.max_events = max_events or self.settings.max_events_per_workflow
        self.max_fix_attempts = max_fix_attempts or self.settings.max_fix_attempts

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
        command_tool = CommandTool(
            workspace,
            timeout_seconds=min(max(self.timeout_seconds, 10.0), 120.0),
        )
        file_awareness = FileAwarenessService(file_tool)
        path_normalizer = FilePathNormalizer(workspace.root)
        content_sanitizer = CodeContentSanitizer()
        fix_target_guard = FixTargetGuard()
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
            path_normalizer,
            content_sanitizer,
            fix_target_guard,
        )

        terminal_inbox = await bus.subscribe_many(
            (
                EventType.TEST_EXECUTION_PASSED if self.allow_execution else EventType.TEST_PASSED,
                EventType.BUILD_PASSED,
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
            terminal_event = await self._wait_for_terminal_event(
                terminal_inbox,
                noise_reducer,
                started_at,
                goal_event.correlation_id or goal_event.id,
            )
            if terminal_event is None:
                terminal_event = Message(
                    sender="runtime",
                    type=EventType.WORKFLOW_TIMEOUT,
                    content={
                        "goal": self.goal,
                        "timeout_seconds": self.timeout_seconds,
                        "reason": "workflow_timeout",
                    },
                    correlation_id=goal_event.correlation_id,
                )
                await bus.publish(terminal_event)

            if terminal_event.type in {
                EventType.TEST_PASSED,
                EventType.TEST_EXECUTION_PASSED,
                EventType.BUILD_PASSED,
            }:
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
                file_awareness,
                content_sanitizer,
                fix_target_guard,
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
        path_normalizer: FilePathNormalizer,
        content_sanitizer: CodeContentSanitizer,
        fix_target_guard: FixTargetGuard,
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
            FileAgent(
                "file_agent",
                bus,
                file_tool,
                graph_store,
                event_logger,
                content_sanitizer,
                fix_target_guard,
            ),
            TesterAgent("tester", bus, file_tool, event_logger),
            TaskCoordinatorAgent(
                "coordinator",
                bus,
                graph_store,
                event_logger,
                max_fix_attempts=self.max_fix_attempts,
                path_normalizer=path_normalizer,
            ),
            SupervisorAgent(
                "supervisor",
                bus,
                event_logger,
                max_events_per_correlation=self.max_events,
                noise_reducer=noise_reducer,
            ),
        ]
        if self.allow_execution:
            agents.insert(
                -2,
                TesterExecutionAgent("tester_execution", bus, command_tool, event_logger),
            )
            agents.insert(
                -2,
                DependencyInstallerAgent(
                    "dependency_installer",
                    bus,
                    command_tool,
                    event_logger,
                ),
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
        file_awareness: FileAwarenessService,
        content_sanitizer: CodeContentSanitizer,
        fix_target_guard: FixTargetGuard,
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
            failure_types, repeated_failures, fixes_attempted = self._fix_summary(graph)
        else:
            failure_types = []
            repeated_failures = []
            fixes_attempted = 0

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
            execution_success_count=self._event_count(
                noise_reducer, EventType.TEST_EXECUTION_PASSED
            ),
            execution_failure_count=self._event_count(
                noise_reducer, EventType.TEST_EXECUTION_FAILED
            ),
            fix_attempts=self._event_count(noise_reducer, EventType.FIX_REQUESTED),
            final_failure_reason=str(dict(terminal_event.content or {}).get("reason", "")),
            detected_failure_types=failure_types,
            fixes_attempted=fixes_attempted,
            repeated_failures=repeated_failures,
            invalid_paths_detected=file_awareness.invalid_paths_detected,
            invalid_paths_ignored=file_awareness.invalid_paths_ignored,
            sanitized_files_count=content_sanitizer.sanitized_files_count,
            wrong_target_fix_count=fix_target_guard.wrong_target_fix_count,
            installed_dependencies=self._event_count(
                noise_reducer,
                EventType.DEPENDENCY_INSTALL_SUCCEEDED,
            ),
            dependency_install_failures=self._event_count(
                noise_reducer,
                EventType.DEPENDENCY_INSTALL_FAILED,
            ),
            patches_applied=self._event_count(noise_reducer, EventType.PATCH_APPLIED),
            patches_failed=self._event_count(noise_reducer, EventType.PATCH_FAILED),
            details=dict(terminal_event.content or {}),
        )

    def _fix_summary(self, graph) -> tuple[list[str], list[str], int]:
        """Summarize fix attempts from the task graph."""
        failure_counter: Counter[str] = Counter()
        fixes_attempted = 0
        for node in graph.nodes.values():
            if node.payload.get("type") != "fix":
                continue
            fixes_attempted += 1
            context = node.payload.get("failure_context", {})
            if isinstance(context, dict):
                failure_type = str(context.get("failure_type", "UnknownFailure"))
                failure_counter[failure_type] += 1
        return (
            sorted(failure_counter),
            sorted(name for name, count in failure_counter.items() if count > 1),
            fixes_attempted,
        )

    async def _wait_for_terminal_event(
        self,
        terminal_inbox: asyncio.Queue[Message],
        noise_reducer: EventNoiseReducer,
        started_at: float,
        correlation_id: str,
    ) -> Message | None:
        """Wait for terminal events while allowing in-flight fixes to finish."""
        while True:
            remaining = self.timeout_seconds - (perf_counter() - started_at)
            if remaining <= 0:
                return None
            try:
                event = await asyncio.wait_for(terminal_inbox.get(), timeout=remaining)
            except TimeoutError:
                return None
            if event.type != EventType.WORKFLOW_HALTED:
                return event
            reason = str(event.content.get("reason", ""))
            if reason == "max_events_exceeded" and self._fix_in_progress(
                noise_reducer,
                correlation_id,
            ):
                continue
            return event

    def _fix_in_progress(self, noise_reducer: EventNoiseReducer, correlation_id: str) -> bool:
        """Return whether fix/retest events indicate an unresolved fix cycle."""
        summary = noise_reducer.summary()
        counts = summary.get("event_counts", {})
        if not isinstance(counts, dict):
            return False
        requested = int(counts.get(str(EventType.FIX_REQUESTED), 0))
        applied = int(counts.get(str(EventType.FIX_APPLIED), 0)) + int(
            counts.get(str(EventType.PATCH_APPLIED), 0)
        )
        retests = int(counts.get(str(EventType.RETEST_REQUESTED), 0))
        passed = int(counts.get(str(EventType.TEST_EXECUTION_PASSED), 0))
        halted = int(counts.get(str(EventType.WORKFLOW_HALTED), 0))
        return (
            requested > 0 and passed == 0 and (requested > applied or applied > retests or halted)
        )

    def _event_count(self, noise_reducer: EventNoiseReducer, event_type: EventType) -> int:
        summary = noise_reducer.summary()
        event_counts = summary.get("event_counts", {})
        if not isinstance(event_counts, dict):
            return 0
        return int(event_counts.get(str(event_type), 0))
