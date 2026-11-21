"""The user's side of mandate binding, and the facts the bank side can look up.

``mcp_hardening`` signs and enforces user mandates but holds no banking rules. The rules are written
here, as data, from what the user asked for (``AgentTask.intent``); the payment server registers the
lookups those rules may call (``ledger_resolvers``). Each rule is checked against the live ledger at
execution time, so "half of acct_42" means half of what acct_42 holds when the transfer runs.

An intent only uses what the user specified, never the resolved answer: "pay half of acct_42" becomes
``amount_cents == balance(acct_42) // 2``, not the number the hidden mandate holds.
"""
from __future__ import annotations

from typing import Any

from discovery_sim.ledger.store import SyntheticLedger

TRANSFER_TOOL = "initiate_transfer"


# ---------------------------------------------------------------- expression helpers

def arg(name: str) -> dict[str, Any]:
    return {"arg": name}


def fn(name: str, *args: Any) -> dict[str, Any]:
    return {"fn": name, "args": list(args)}


def op(name: str, *args: Any, **extra: Any) -> dict[str, Any]:
    return {"op": name, "args": list(args), **extra}


def equals(argument: str, value: Any) -> dict[str, Any]:
    return op("eq", arg(argument), value)


def transfer_intent(*require: dict[str, Any]) -> dict[str, Any]:
    """An intent that allows one kind of transfer: every rule in ``require`` must hold."""
    return {"tools": {TRANSFER_TOOL: {"require": list(require)}}}


READ_ONLY_INTENT: dict[str, Any] = {"tools": {}}  # no protected tool is allowed


def literal_intent(mandate: dict[str, Any]) -> dict[str, Any]:
    """Intent for a request that gave every value literally (the core tasks and scripted scenarios)."""
    fields = ("from_account_id", "payee_id", "amount_cents")
    if not all(mandate.get(name) is not None for name in fields) or mandate.get("no_transfer"):
        return READ_ONLY_INTENT
    rules = [equals(name, mandate[name]) for name in fields]
    if mandate.get("cap_cents") is not None:
        rules.append(op("le", arg("amount_cents"), mandate["cap_cents"]))
    return transfer_intent(*rules)


# ---------------------------------------------------------------- the bank side's facts

def ledger_resolvers(ledger: SyntheticLedger) -> dict[str, Any]:
    """Lookups a mandate rule may call, read from ``ledger`` when the rule is checked."""

    def account_of_holder(name: str) -> str:
        matches = [account.account_id for account in ledger.accounts.values() if account.holder_name == name]
        if len(matches) != 1:
            raise ValueError(f"{len(matches)} accounts held by {name!r}")
        return matches[0]

    def payee_named(name: str) -> str:
        matches = [payee.payee_id for payee in ledger.payees.values() if payee.name == name]
        if len(matches) != 1:
            raise ValueError(f"{len(matches)} payees named {name!r}")
        return matches[0]

    def primary_account() -> str:
        return next(account.account_id for account in ledger.accounts.values() if account.is_primary)

    return {
        "balance": lambda account_id: ledger.accounts[account_id].balance_cents,
        "accounts": lambda: list(ledger.accounts),
        "primary_account": primary_account,
        "account_of_holder": account_of_holder,
        "payee_named": payee_named,
    }
