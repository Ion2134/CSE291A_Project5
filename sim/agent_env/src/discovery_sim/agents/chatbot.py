from __future__ import annotations

import uuid

from discovery_sim.a2a.types import (
    AgentSkill,
    SendMessageRequest,
    SendMessageResponse,
    completed_task,
    extract_text_from_message,
    skill_id_from_message,
)
from discovery_sim.agents.base import A2AAgent


class ChatbotAgent(A2AAgent):
    """User-facing orchestrator; LLM wiring comes later."""

    agent_name = "Synth Chatbot"
    agent_description = "Orchestrates banking tasks over local A2A peers."
    skills = [
        AgentSkill(
            id="chat.answer",
            name="Answer",
            description="Accept a user message and return an assistant reply.",
            tags=["chat", "orchestration"],
            examples=["What is my balance on acct_synth_0031?"],
        ),
    ]

    async def send_message(self, request: SendMessageRequest) -> SendMessageResponse:
        skill_id = skill_id_from_message(request.message) or "chat.answer"
        if skill_id != "chat.answer":
            raise ValueError(f"unsupported skill: {skill_id}")
        user_text = extract_text_from_message(request.message)
        if not user_text.strip():
            raise ValueError("chat.answer requires a text part")
        task_id = str(uuid.uuid4())
        reply = (
            "Chatbot stub: I received your message. "
            "LLM orchestration is not wired yet; call bank or payment agents directly."
        )
        return SendMessageResponse(
            task=completed_task(task_id, {"reply": reply, "echo": user_text.strip()}),
        )
