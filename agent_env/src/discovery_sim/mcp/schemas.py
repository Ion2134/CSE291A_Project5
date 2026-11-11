from __future__ import annotations

from typing import Any


def _tool(name: str, description: str, input_schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "inputSchema": input_schema,
    }


BANK_TOOLS: list[dict[str, Any]] = [
    _tool(
        "get_balance",
        "Return the current balance for a synthetic account.",
        {
            "type": "object",
            "properties": {"account_id": {"type": "string"}},
            "required": ["account_id"],
            "additionalProperties": False,
        },
    ),
    _tool(
        "list_accounts",
        "List synthetic accounts for the tenant.",
        {"type": "object", "properties": {}, "additionalProperties": False},
    ),
    _tool(
        "get_transactions",
        "List recent transactions for an account.",
        {
            "type": "object",
            "properties": {
                "account_id": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": ["account_id"],
            "additionalProperties": False,
        },
    ),
    _tool(
        "get_account_holder",
        "Return holder metadata for an account.",
        {
            "type": "object",
            "properties": {"account_id": {"type": "string"}},
            "required": ["account_id"],
            "additionalProperties": False,
        },
    ),
    _tool(
        "list_payees",
        "List the tenant's registered payees (payee id and name).",
        {"type": "object", "properties": {}, "additionalProperties": False},
    ),
]

PAYMENT_TOOLS: list[dict[str, Any]] = [
    _tool(
        "initiate_transfer",
        "Transfer funds from an account to a payee.",
        {
            "type": "object",
            "properties": {
                "from_account_id": {"type": "string"},
                "payee_id": {"type": "string"},
                "amount_cents": {"type": "integer", "minimum": 1},
            },
            "required": ["from_account_id", "payee_id", "amount_cents"],
            "additionalProperties": False,
        },
    ),
    _tool(
        "get_payment_status",
        "Return status for a payment id.",
        {
            "type": "object",
            "properties": {"payment_id": {"type": "string"}},
            "required": ["payment_id"],
            "additionalProperties": False,
        },
    ),
    _tool(
        "cancel_payment",
        "Cancel a completed payment when supported.",
        {
            "type": "object",
            "properties": {"payment_id": {"type": "string"}},
            "required": ["payment_id"],
            "additionalProperties": False,
        },
    ),
    _tool(
        "add_payee",
        "Register a new synthetic payee.",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "routing_hint": {"type": "string"},
            },
            "required": ["name", "routing_hint"],
            "additionalProperties": False,
        },
    ),
]
