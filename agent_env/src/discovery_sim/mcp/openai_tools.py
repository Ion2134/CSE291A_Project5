from __future__ import annotations

from typing import Any, Literal

from discovery_sim.mcp.schemas import BANK_TOOLS, PAYMENT_TOOLS

AgentName = Literal["bank", "payment"]


def _openai_tool(agent: AgentName, spec: dict[str, Any]) -> dict[str, Any]:
    name = f"{agent}__{spec['name']}"
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"[{agent}] {spec['description']}",
            "parameters": spec["inputSchema"],
        },
    }


def all_openai_tools() -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for spec in BANK_TOOLS:
        tools.append(_openai_tool("bank", spec))
    for spec in PAYMENT_TOOLS:
        tools.append(_openai_tool("payment", spec))
    return tools


def parse_tool_name(openai_name: str) -> tuple[AgentName, str]:
    if "__" not in openai_name:
        raise ValueError(f"invalid tool name: {openai_name}")
    agent, tool = openai_name.split("__", 1)
    if agent not in ("bank", "payment"):
        raise ValueError(f"unknown agent in tool: {openai_name}")
    return agent, tool  # type: ignore[return-value]
