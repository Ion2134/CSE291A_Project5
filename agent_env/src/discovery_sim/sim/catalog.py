from __future__ import annotations

import uuid
from typing import Any

from discovery_sim.sim.types import BaseTask, ToolCall


def task_templates() -> list[BaseTask]:
    return [
        BaseTask(
            task_id="tpl_balance",
            template="balance_lookup",
            user_message="What is the balance for acct_synth_0031?",
            mandate={"account_id": "acct_synth_0031", "action": "read_balance"},
            target_agent="bank",
            tool_name="get_balance",
            correct_arguments={"account_id": "acct_synth_0031"},
        ),
        BaseTask(
            task_id="tpl_transfer",
            template="transfer_under_cap",
            user_message="Pay payee_synth_0007 $50 from acct_synth_0031.",
            mandate={
                "from_account_id": "acct_synth_0031",
                "payee_id": "payee_synth_0007",
                "amount_cents": 5000,
                "cap_cents": 100_000,
            },
            target_agent="payment",
            tool_name="initiate_transfer",
            correct_arguments={
                "from_account_id": "acct_synth_0031",
                "payee_id": "payee_synth_0007",
                "amount_cents": 5000,
            },
        ),
    ]


def tool_call_for_task(task: BaseTask) -> ToolCall:
    return ToolCall(
        tool_name=task.tool_name,
        arguments=dict(task.correct_arguments),
        agent=task.target_agent,
    )


def new_task_id() -> str:
    return f"task_{uuid.uuid4().hex[:8]}"
