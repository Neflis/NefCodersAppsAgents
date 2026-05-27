"""Demo entrypoint for the async multi-agent lab."""

from __future__ import annotations

import asyncio

from multi_agent_lab.agents.coder_agent import CoderAgent
from multi_agent_lab.agents.planner_agent import PlannerAgent
from multi_agent_lab.agents.reviewer_agent import ReviewerAgent
from multi_agent_lab.core.message_bus import MessageBus
from multi_agent_lab.core.task_queue import TaskQueue


async def run_demo() -> None:
    bus = MessageBus()
    task_queue = TaskQueue()

    planner = PlannerAgent("planner", bus, task_queue)
    coder = CoderAgent("coder", bus, task_queue)
    reviewer = ReviewerAgent("reviewer", bus)
    agents = [planner, coder, reviewer]

    for agent in agents:
        await agent.start()

    await planner.create_example_task()
    await asyncio.sleep(0.5)

    for agent in agents:
        await agent.stop()


def main() -> None:
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
