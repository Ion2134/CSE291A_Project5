from __future__ import annotations

from typing import Any, Callable

from discovery_sim.ledger.store import SyntheticLedger


class McpToolServer:
    """Local MCP-shaped tool dispatcher backed by the synthetic ledger."""

    def __init__(
        self,
        tools: list[dict[str, Any]],
        handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]],
    ) -> None:
        self._tools = tools
        self._handlers = handlers

    def tools_list(self) -> dict[str, Any]:
        return {"tools": self._tools}

    def tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(name, str) or not name:
            raise ValueError("tools/call requires name")
        if not isinstance(arguments, dict):
            raise ValueError("tools/call arguments must be an object")
        handler = self._handlers.get(name)
        if handler is None:
            raise ValueError(f"unknown tool: {name}")
        content = handler(arguments)
        return {
            "content": [{"type": "text", "text": _json_text(content)}],
            "structuredContent": content,
            "isError": False,
        }


def _json_text(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, separators=(",", ":"))


def bank_mcp_server(ledger: SyntheticLedger) -> McpToolServer:
    from discovery_sim.mcp.schemas import BANK_TOOLS

    handlers = {
        "get_balance": lambda args: ledger.get_balance(args["account_id"]),
        "list_accounts": lambda _: {"accounts": ledger.list_accounts()},
        "get_transactions": lambda args: ledger.get_transactions(
            args["account_id"], int(args.get("limit") or 10)
        ),
        "get_account_holder": lambda args: ledger.get_account_holder(args["account_id"]),
        "list_payees": lambda _: {"payees": ledger.list_payees()},
    }
    return McpToolServer(BANK_TOOLS, handlers)


def payment_mcp_server(ledger: SyntheticLedger) -> McpToolServer:
    from discovery_sim.mcp.schemas import PAYMENT_TOOLS

    handlers = {
        "initiate_transfer": lambda args: ledger.initiate_transfer(
            args["from_account_id"], args["payee_id"], int(args["amount_cents"])
        ),
        "get_payment_status": lambda args: ledger.get_payment_status(args["payment_id"]),
        "cancel_payment": lambda args: ledger.cancel_payment(args["payment_id"]),
        "add_payee": lambda args: ledger.add_payee(args["name"], args["routing_hint"]),
    }
    return McpToolServer(PAYMENT_TOOLS, handlers)
