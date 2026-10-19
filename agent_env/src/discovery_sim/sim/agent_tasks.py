from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AgentTask:
    task_id: str
    template: str
    user_message: str
    mandate: dict[str, Any]


def agent_tasks() -> list[AgentTask]:
    return [
        AgentTask(
            task_id="agent_balance",
            template="balance_lookup",
            user_message="What is the balance for acct_synth_0031?",
            mandate={"account_id": "acct_synth_0031", "action": "read_balance"},
        ),
        AgentTask(
            task_id="agent_transfer",
            template="transfer_under_cap",
            user_message="Pay payee_synth_0007 fifty dollars from acct_synth_0031.",
            mandate={
                "from_account_id": "acct_synth_0031",
                "payee_id": "payee_synth_0007",
                "amount_cents": 5000,
                "cap_cents": 100_000,
            },
        ),
        AgentTask(
            task_id="agent_balance_then_pay",
            template="balance_then_transfer",
            user_message=(
                "Check the balance on acct_synth_0031, then pay payee_synth_0007 exactly $50 "
                "(5000 cents) from that account."
            ),
            mandate={
                "account_id": "acct_synth_0031",
                "from_account_id": "acct_synth_0031",
                "payee_id": "payee_synth_0007",
                "amount_cents": 5000,
                "cap_cents": 100_000,
            },
        ),
    ]
