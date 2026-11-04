from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from discovery_sim.ledger.store import SyntheticLedger
from discovery_sim.sim.types import CallRecord, ToolCall


@dataclass
class PriorCallSummary:
    call_index: int
    llm_call: ToolCall
    received_args: dict[str, Any]
    output: dict[str, Any]
    output_correct: bool | None = None


@dataclass
class AgentContext:
    """Local agent state plus a view of global ledger for one episode."""

    user_message: str
    mandate: dict[str, Any]
    trusted: dict[str, Any]
    prior_calls: list[PriorCallSummary] = field(default_factory=list)
    taint_map: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_mandate(cls, user_message: str, mandate: dict[str, Any]) -> AgentContext:
        return cls(user_message=user_message, mandate=dict(mandate), trusted=dict(mandate))

    def record_call(self, index: int, record: CallRecord, output_correct: bool | None) -> None:
        self.prior_calls.append(
            PriorCallSummary(
                call_index=index,
                llm_call=record.llm_call,
                received_args=dict(record.mcp_arguments),
                output=dict(record.output),
                output_correct=output_correct,
            )
        )

    def ledger_snapshot(self, ledger: SyntheticLedger) -> dict[str, Any]:
        # Keep this small: full account dicts blow Laya sequence length on MPS.
        return {
            "balances": {account_id: account.balance_cents for account_id, account in ledger.accounts.items()},
            "payee_count": len(ledger.payees),
            "payment_count": len(ledger.payments),
        }
