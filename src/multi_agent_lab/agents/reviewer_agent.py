"""Reviewer agent that reviews simulated coder output."""

from __future__ import annotations

from multi_agent_lab.agents.base_agent import BaseAgent
from multi_agent_lab.core.message import Message


class ReviewerAgent(BaseAgent):
    async def handle_message(self, message: Message) -> None:
        if message.type != "code.simulated":
            print(f"[reviewer] mensaje ignorado: {message.type}")
            return

        content = message.content
        print(f"[reviewer] revisando tarea {content['task_id']}")
        review = {
            "task_id": content["task_id"],
            "approved": True,
            "notes": "Respuesta simulada aceptada para la demo asíncrona.",
        }
        await self.publish("planner", "review.completed", review)
