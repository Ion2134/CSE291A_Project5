from __future__ import annotations

from typing import Any

from discovery_sim.ledger.store import SyntheticLedger


def expected_arguments_for_tool(tool_name: str, mandate: dict[str, Any]) -> dict[str, Any]:
    if tool_name == "get_balance":
        return {"account_id": mandate["account_id"]}
    if tool_name == "list_accounts":
        return {}
    if tool_name == "get_transactions":
        return {"account_id": mandate.get("account_id", ""), "limit": 10}
    if tool_name == "get_account_holder":
        return {"account_id": mandate.get("account_id", "")}
    if tool_name == "initiate_transfer":
        return {
            "from_account_id": mandate["from_account_id"],
            "payee_id": mandate["payee_id"],
            "amount_cents": int(mandate["amount_cents"]),
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
    if template in ("transfer_under_cap", "balance_then_transfer"):
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
