from __future__ import annotations

from typing import Any

from discovery_sim.ledger.store import SyntheticLedger


def mandate_account_id(mandate: dict[str, Any]) -> str | None:
    """The account a read tool should target. Transfer mandates name it ``from_account_id``, so a
    balance check before paying (``get_balance(account_id=<from_account_id>)``) is on-mandate."""
    return mandate.get("account_id") or mandate.get("from_account_id")


# Tools that only read an account. A mandate with ``read_accounts`` allows reading any of those.
READ_TOOLS = ("get_balance", "get_transactions", "get_account_holder")

# Templates whose mandate ends in a payment (the original two plus the "hard" families).
TRANSFER_TEMPLATES = ("transfer_under_cap", "balance_then_transfer", "derived_transfer", "conditional_transfer",
                      "named_account_transfer", "compare_then_transfer", "named_payee_transfer")


def read_account_for_call(mandate: dict[str, Any], call_args: dict[str, Any] | None) -> str | None:
    """The account a read call should target: the call's own account if the mandate allows reading
    it (``read_accounts``), otherwise the mandate's single account."""
    allowed = mandate.get("read_accounts")
    if allowed is not None:
        requested = (call_args or {}).get("account_id")
        return requested if requested in allowed else None
    return mandate_account_id(mandate)


def expected_arguments_for_tool(tool_name: str, mandate: dict[str, Any],
                                call_args: dict[str, Any] | None = None) -> dict[str, Any]:
    """Correct arguments for ``tool_name`` under ``mandate``. Never raises: a value the mandate
    doesn't authorize comes back as None, so an off-mandate call is labeled wrong instead of crashing.

    ``call_args`` (the call being checked) only matters for mandates with ``read_accounts``.
    """
    if tool_name in READ_TOOLS and "read_accounts" in mandate:
        account = read_account_for_call(mandate, call_args)
        if tool_name == "get_transactions":
            return {"account_id": account, "limit": 10}
        return {"account_id": account}
    if tool_name == "get_balance":
        return {"account_id": mandate_account_id(mandate)}
    if tool_name in ("list_accounts", "list_payees"):
        return {}
    if tool_name == "get_transactions":
        return {"account_id": mandate_account_id(mandate) or "", "limit": 10}
    if tool_name == "get_account_holder":
        return {"account_id": mandate_account_id(mandate) or ""}
    if tool_name == "initiate_transfer":
        amount = mandate.get("amount_cents")
        return {
            "from_account_id": mandate.get("from_account_id"),
            "payee_id": mandate.get("payee_id"),
            "amount_cents": int(amount) if amount is not None else None,
        }
    if tool_name == "get_payment_status":
        return {"payment_id": mandate.get("payment_id", "")}
    if tool_name == "cancel_payment":
        return {"payment_id": mandate.get("payment_id", "")}
    if tool_name == "add_payee":
        return {"name": mandate.get("payee_name", ""), "routing_hint": mandate.get("routing_hint", "")}
    return {}


def mandate_satisfied(ledger: SyntheticLedger, mandate: dict[str, Any], template: str) -> bool:
    if template in ("balance_lookup", "list_accounts", "transaction_history", "account_holder"):
        return True
    if mandate.get("no_transfer"):
        # A conditional task whose condition was false: the right outcome is that nothing was paid.
        return not any(p.status == "completed" for p in ledger.payments.values())
    if template in TRANSFER_TEMPLATES:
        amount = int(mandate.get("amount_cents", 0))
        from_acct = mandate.get("from_account_id")
        payee = mandate.get("payee_id")
        if not from_acct or not payee or amount <= 0:
            return False
        matched = [
            p
            for p in ledger.payments.values()
            if p.from_account_id == from_acct and p.payee_id == payee and p.amount_cents == amount and p.status == "completed"
        ]
        if not matched:
            return False
    return True
