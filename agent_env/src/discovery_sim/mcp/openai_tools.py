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


_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
}


def _schemas_by_openai_name() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for agent, specs in (("bank", BANK_TOOLS), ("payment", PAYMENT_TOOLS)):
        for spec in specs:
            out[f"{agent}__{spec['name']}"] = spec["inputSchema"]
    return out


def validate_tool_call(openai_name: str, arguments: dict[str, Any]) -> str | None:
    """Check a model tool call against the tool schemas. Returns an error message, or None if valid.

    Catches calls the model malformed (unknown tool, missing/extra/mistyped arguments) before they
    reach the simulator, so they're reported as model failures rather than simulator crashes.
    """
    schema = _schemas_by_openai_name().get(openai_name)
    if schema is None:
        return f"unknown tool: {openai_name!r}"
    props: dict[str, Any] = schema.get("properties", {})
    missing = [key for key in schema.get("required", []) if key not in arguments]
    if missing:
        return f"missing required argument(s): {', '.join(missing)}"
    if schema.get("additionalProperties") is False:
        extra = sorted(set(arguments) - set(props))
        if extra:
            return f"unexpected argument(s): {', '.join(extra)}"
    for key, value in arguments.items():
        spec = props.get(key, {})
        expected = spec.get("type")
        # bool is a subclass of int in Python, but not a JSON integer.
        if expected in _JSON_TYPES and (
            not isinstance(value, _JSON_TYPES[expected]) or (expected in ("integer", "number") and isinstance(value, bool))
        ):
            return f"argument {key!r} must be {expected}, got {type(value).__name__}"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in spec and value < spec["minimum"]:
                return f"argument {key!r} must be >= {spec['minimum']}, got {value}"
            if "maximum" in spec and value > spec["maximum"]:
                return f"argument {key!r} must be <= {spec['maximum']}, got {value}"
    return None
