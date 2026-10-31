from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from starlette.applications import Starlette

from discovery_sim.a2a.server import create_a2a_app
from discovery_sim.a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    SendMessageRequest,
    SendMessageResponse,
)
from discovery_sim.mcp.server import McpToolServer


class A2AAgent(ABC):
    """Base class for local A2A 1.0 agents."""

    agent_name: str
    agent_description: str
    agent_version: str = "1.0.0"
    skills: list[AgentSkill]

    def __init__(self, public_base_url: str) -> None:
        self.public_base_url = public_base_url.rstrip("/")

    def build_card(self) -> AgentCard:
        return AgentCard(
            name=self.agent_name,
            description=self.agent_description,
            version=self.agent_version,
            supported_interfaces=[
                AgentInterface(url=f"{self.public_base_url}/rpc"),
            ],
            capabilities=AgentCapabilities(),
            default_input_modes=["text/plain", "application/json"],
            default_output_modes=["text/plain", "application/json"],
            skills=self.skills,
        )

    @abstractmethod
    async def send_message(self, request: SendMessageRequest) -> SendMessageResponse:
        raise NotImplementedError

    def mcp_server(self) -> McpToolServer | None:
        return None

    def starlette_app(self) -> Starlette:
        card = self.build_card()
        mcp = self.mcp_server()
        if mcp is None:
            return create_a2a_app(card, self.send_message)
        return create_a2a_app(
            card,
            self.send_message,
            mcp_tools_list=mcp.tools_list,
            mcp_tools_call=self._mcp_call,
        )

    async def _mcp_call(self, params: dict[str, Any]) -> dict[str, Any]:
        mcp = self.mcp_server()
        if mcp is None:
            raise ValueError("no MCP server")
        return mcp.tools_call(params)


def skills_from_mcp_tools(tools: list[dict[str, Any]], tag: str) -> list[AgentSkill]:
    return [
        AgentSkill(
            id=t["name"],
            name=t["name"].replace("_", " ").title(),
            description=t["description"],
            tags=[tag, "mcp"],
        )
        for t in tools
    ]


class McpBackedAgent(A2AAgent):
    """Agent whose A2A skills map 1:1 to MCP tool names."""

    def __init__(self, public_base_url: str, mcp: McpToolServer, skill_tag: str) -> None:
        super().__init__(public_base_url)
        self._mcp = mcp
        self.skills = skills_from_mcp_tools(mcp.tools_list()["tools"], skill_tag)

    def mcp_server(self) -> McpToolServer | None:
        return self._mcp

    async def send_message(self, request: SendMessageRequest) -> SendMessageResponse:
        import uuid

        from discovery_sim.a2a.types import (
            SendMessageResponse as Response,
            completed_task,
            extract_json_from_message,
            skill_id_from_message,
        )

        skill_id = skill_id_from_message(request.message)
        if not skill_id:
            raise ValueError("message.metadata.skillId is required")
        arguments = extract_json_from_message(request.message) or {}
        result = self._mcp.tools_call({"name": skill_id, "arguments": arguments})
        payload = result.get("structuredContent") or {}
        task_id = str(uuid.uuid4())
        return Response(task=completed_task(task_id, payload))
