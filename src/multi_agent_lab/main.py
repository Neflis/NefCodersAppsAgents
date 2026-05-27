"""Demo entrypoint for the async multi-agent lab."""

from __future__ import annotations

import argparse
import asyncio
import logging

from multi_agent_lab.agents.coder_agent import CoderAgent
from multi_agent_lab.agents.planner_agent import PlannerAgent
from multi_agent_lab.agents.reviewer_agent import ReviewerAgent
from multi_agent_lab.config.settings import load_settings
from multi_agent_lab.core.agent_event_logger import AgentEventLogger
from multi_agent_lab.core.message_bus import MessageBus
from multi_agent_lab.core.sqlite_store import SQLiteStore
from multi_agent_lab.core.task_queue import TaskQueue
from multi_agent_lab.llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


async def run_demo(mode: str = "demo_mock") -> None:
    """Run the local multi-agent demo."""
    settings = load_settings()
    store = SQLiteStore(settings.database_url)
    event_logger = AgentEventLogger(store)
    bus = MessageBus(store)
    task_queue = TaskQueue(store)

    ollama_client = OllamaClient(settings.ollama_base_url, settings.ollama_model)
    use_ollama = mode == "demo_ollama"
    if use_ollama and not await ollama_client.health_check():
        logger.info("Ollama no disponible; se usara respuesta local simulada.")
        use_ollama = False

    planner = PlannerAgent("planner", bus, task_queue, event_logger)
    coder = CoderAgent("coder", bus, task_queue, event_logger, ollama_client, use_ollama)
    reviewer = ReviewerAgent("reviewer", bus, event_logger)
    agents = [planner, coder, reviewer]

    try:
        for agent in agents:
            await agent.start()

        await planner.create_example_task()
        await asyncio.sleep(0.5)
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
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    asyncio.run(run_demo(args.mode))


if __name__ == "__main__":
    main()
