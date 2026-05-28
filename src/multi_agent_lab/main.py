"""Demo entrypoint for the async multi-agent lab."""

from __future__ import annotations

import argparse
import asyncio
import logging

from multi_agent_lab.agents.coder_agent import CoderAgent
from multi_agent_lab.agents.file_agent import FileAgent
from multi_agent_lab.agents.planner_agent import PlannerAgent
from multi_agent_lab.agents.reviewer_agent import ReviewerAgent
from multi_agent_lab.agents.supervisor_agent import SupervisorAgent
from multi_agent_lab.agents.task_coordinator_agent import TaskCoordinatorAgent
from multi_agent_lab.agents.tester_agent import TesterAgent
from multi_agent_lab.config.settings import load_settings
from multi_agent_lab.core.agent_event_logger import AgentEventLogger
from multi_agent_lab.core.message import EventType, Message
from multi_agent_lab.core.message_bus import MessageBus
from multi_agent_lab.core.sqlite_store import SQLiteStore
from multi_agent_lab.core.task_graph_store import TaskGraphStore
from multi_agent_lab.core.workspace_manager import WorkspaceManager
from multi_agent_lab.llm.context_builder import AgentContextBuilder
from multi_agent_lab.llm.ollama_client import OllamaClient
from multi_agent_lab.tools.file_tool import FileTool

logger = logging.getLogger(__name__)


async def run_demo(mode: str = "demo_mock") -> None:
    """Run the autonomous task graph demo."""
    settings = load_settings()
    store = SQLiteStore(settings.database_url)
    event_logger = AgentEventLogger(store)
    bus = MessageBus(store)
    graph_store = TaskGraphStore(store)
    workspace = WorkspaceManager("workspace")
    file_tool = FileTool(workspace)

    planner_llm = OllamaClient(
        settings.ollama_base_url,
        settings.ollama_model_planner,
        settings.ollama_timeout_seconds,
        use_mock=settings.use_mock_llm,
    )
    coder_llm = OllamaClient(
        settings.ollama_base_url,
        settings.ollama_model_coder,
        settings.ollama_timeout_seconds,
        use_mock=settings.use_mock_llm,
    )
    reviewer_llm = OllamaClient(
        settings.ollama_base_url,
        settings.ollama_model_reviewer,
        settings.ollama_timeout_seconds,
        use_mock=settings.use_mock_llm,
    )
    use_ollama = mode == "demo_ollama"
    if use_ollama and not await coder_llm.health_check():
        logger.info("Ollama no disponible; se usara respuesta local simulada.")
        use_ollama = False
    if not use_ollama:
        planner_llm.use_mock = True
        coder_llm.use_mock = True
        reviewer_llm.use_mock = True

    context_builder = AgentContextBuilder(graph_store, file_tool)

    test_results = await bus.subscribe(EventType.TEST_PASSED)
    agents = [
        PlannerAgent("planner", bus, graph_store, event_logger, planner_llm, context_builder),
        CoderAgent("coder", bus, event_logger, coder_llm, context_builder),
        ReviewerAgent("reviewer", bus, graph_store, event_logger, reviewer_llm, context_builder),
        FileAgent("file_agent", bus, file_tool, graph_store, event_logger),
        TesterAgent("tester", bus, file_tool, event_logger),
        TaskCoordinatorAgent("coordinator", bus, graph_store, event_logger),
        SupervisorAgent("supervisor", bus, event_logger),
    ]

    try:
        for agent in agents:
            await agent.start()

        await bus.publish(
            Message(
                sender="demo",
                type=EventType.GOAL_SUBMITTED,
                content={
                    "goal": "Crear una pequena documentacion README para una app TODO",
                    "path": "README.md",
                },
                metadata={"mode": mode},
            )
        )
        await asyncio.wait_for(test_results.get(), timeout=3)
    finally:
        for agent in agents:
            await agent.stop()
        store.close()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the async multi-agent demo.")
    parser.add_argument(
        "--mode",
        choices=["demo_mock", "demo_ollama"],
        default="demo_mock",
        help="demo_mock avoids Ollama calls; demo_ollama uses Ollama when available.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the command-line entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    )
    args = parse_args()
    asyncio.run(run_demo(args.mode))


if __name__ == "__main__":
    main()
