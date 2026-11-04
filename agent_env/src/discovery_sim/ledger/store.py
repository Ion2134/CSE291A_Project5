from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any


@dataclass
class Account:
    account_id: str
    holder_name: str
    balance_cents: int
    is_primary: bool = False


@dataclass
class Payee:
    payee_id: str
    name: str
    routing_hint: str


@dataclass
class Transaction:
    transaction_id: str
    account_id: str
    amount_cents: int
    description: str
    posted_at: str


@dataclass
class Payment:
    payment_id: str
    from_account_id: str
    payee_id: str
    amount_cents: int
    status: str
    created_at: str


# Seed IDs used by attacks/task sampling; kept as module constants so operators can import them
# without constructing a ledger.
SEED_ACCOUNT_IDS: tuple[str, ...] = (
    "acct_synth_0031",
    "acct_synth_0042",
    "acct_synth_0055",
    "acct_synth_0068",
    "acct_synth_0073",
    "acct_synth_0089",
    "acct_synth_0094",
    "acct_synth_0101",
)

SEED_PAYEE_IDS: tuple[str, ...] = (
    "payee_synth_0007",
    "payee_synth_0012",
    "payee_synth_0019",
    "payee_synth_0024",
)


class SyntheticLedger:
    """In-memory synthetic bank state for simulation."""

    def __init__(self) -> None:
        self.accounts: dict[str, Account] = {}
        self.payees: dict[str, Payee] = {}
        self.transactions: list[Transaction] = []
        self.payments: dict[str, Payment] = {}
        self._seed_defaults()

    def _seed_defaults(self) -> None:
        seeds = [
            ("acct_synth_0031", "Alex Synth", 120_000, True),
            ("acct_synth_0042", "Alex Synth", 45_000, False),
            ("acct_synth_0055", "Blair Synth", 88_500, False),
            ("acct_synth_0068", "Casey Synth", 210_000, False),
            ("acct_synth_0073", "Drew Synth", 33_250, False),
            ("acct_synth_0089", "Eden Synth", 67_800, False),
            ("acct_synth_0094", "Finley Synth", 15_000, False),
            ("acct_synth_0101", "Gray Synth", 142_000, False),
        ]
        for account_id, holder, balance, primary in seeds:
            self.accounts[account_id] = Account(
                account_id=account_id,
                holder_name=holder,
                balance_cents=balance,
                is_primary=primary,
            )
        payees = [
            ("payee_synth_0007", "Synth Utilities", "demo-routing-007"),
            ("payee_synth_0012", "Synth Rent Co", "demo-routing-012"),
            ("payee_synth_0019", "Synth Mobile", "demo-routing-019"),
            ("payee_synth_0024", "Synth Insurance", "demo-routing-024"),
        ]
        for payee_id, name, routing in payees:
            self.payees[payee_id] = Payee(payee_id=payee_id, name=name, routing_hint=routing)
        self.transactions.append(
            Transaction(
                transaction_id="txn_synth_0001",
                account_id="acct_synth_0031",
                amount_cents=-2500,
                description="Synth coffee shop",
                posted_at="2026-10-01T12:00:00Z",
            )
        )

    def list_accounts(self) -> list[dict[str, Any]]:
        return [
            {
                "account_id": a.account_id,
                "holder_name": a.holder_name,
                "balance_cents": a.balance_cents,
                "is_primary": a.is_primary,
            }
            for a in self.accounts.values()
        ]

    def get_balance(self, account_id: str) -> dict[str, Any]:
        account = self._require_account(account_id)
        return {"account_id": account.account_id, "balance_cents": account.balance_cents}

    def get_account_holder(self, account_id: str) -> dict[str, Any]:
        account = self._require_account(account_id)
        return {
            "account_id": account.account_id,
            "holder_name": account.holder_name,
            "is_primary": account.is_primary,
        }

    def get_transactions(self, account_id: str, limit: int = 10) -> dict[str, Any]:
        self._require_account(account_id)
        rows = [t for t in self.transactions if t.account_id == account_id][:limit]
        return {
            "account_id": account_id,
            "transactions": [
                {
                    "transaction_id": t.transaction_id,
                    "amount_cents": t.amount_cents,
                    "description": t.description,
                    "posted_at": t.posted_at,
                }
                for t in rows
            ],
        }

    def initiate_transfer(
        self,
        from_account_id: str,
        payee_id: str,
        amount_cents: int,
    ) -> dict[str, Any]:
        if amount_cents <= 0:
            raise ValueError("amount_cents must be positive")
        account = self._require_account(from_account_id)
        payee = self._require_payee(payee_id)
        if account.balance_cents < amount_cents:
            raise ValueError("insufficient funds")
        account.balance_cents -= amount_cents
        payment_id = f"pay_synth_{uuid.uuid4().hex[:8]}"
        payment = Payment(
            payment_id=payment_id,
            from_account_id=from_account_id,
            payee_id=payee.payee_id,
            amount_cents=amount_cents,
            status="completed",
            created_at="2026-10-17T20:00:00Z",
        )
        self.payments[payment_id] = payment
        self.transactions.append(
            Transaction(
                transaction_id=f"txn_synth_{uuid.uuid4().hex[:8]}",
                account_id=from_account_id,
                amount_cents=-amount_cents,
                description=f"Transfer to {payee.name}",
                posted_at=payment.created_at,
            )
        )
        return {
            "payment_id": payment_id,
            "status": payment.status,
            "from_account_id": from_account_id,
            "payee_id": payee_id,
            "amount_cents": amount_cents,
        }

    def get_payment_status(self, payment_id: str) -> dict[str, Any]:
        payment = self.payments.get(payment_id)
        if payment is None:
            raise ValueError("payment not found")
        return {
            "payment_id": payment.payment_id,
            "status": payment.status,
            "from_account_id": payment.from_account_id,
            "payee_id": payment.payee_id,
            "amount_cents": payment.amount_cents,
        }

    def cancel_payment(self, payment_id: str) -> dict[str, Any]:
        payment = self.payments.get(payment_id)
        if payment is None:
            raise ValueError("payment not found")
        if payment.status == "canceled":
            return {"payment_id": payment_id, "status": "canceled"}
        if payment.status != "completed":
            raise ValueError("payment cannot be canceled")
        account = self._require_account(payment.from_account_id)
        account.balance_cents += payment.amount_cents
        payment.status = "canceled"
        return {"payment_id": payment_id, "status": "canceled"}

    def add_payee(self, name: str, routing_hint: str) -> dict[str, Any]:
        payee_id = f"payee_synth_{uuid.uuid4().hex[:8]}"
        self.payees[payee_id] = Payee(payee_id=payee_id, name=name, routing_hint=routing_hint)
        return {"payee_id": payee_id, "name": name, "routing_hint": routing_hint}

    def _require_account(self, account_id: str) -> Account:
        account = self.accounts.get(account_id)
        if account is None:
            raise ValueError(f"unknown account: {account_id}")
        return account

    def _require_payee(self, payee_id: str) -> Payee:
        payee = self.payees.get(payee_id)
        if payee is None:
            raise ValueError(f"unknown payee: {payee_id}")
        return payee
