from __future__ import annotations

import random

from discovery_sim.sim.agent_tasks import AgentTask
from discovery_sim.sim.types import AttackSpec, OperatorId


_BANK_TEMPLATES = frozenset(
    {"balance_lookup", "list_accounts", "transaction_history", "account_holder"}
)


def _plan_rounds(task: AgentTask, prefix: str) -> list[int]:
    """Rounds of the task's planned calls whose tool name starts with ``prefix`` ("" for all)."""
    return [index for index, (name, _) in enumerate(task.plan) if name.startswith(prefix)]


def _expected_rounds_for_task(task: AgentTask) -> list[int]:
    if task.plan is not None:
        return _plan_rounds(task, "")
    if task.template == "balance_then_transfer":
        return [0, 1]
    return [0]


def _bank_rounds(task: AgentTask) -> list[int]:
    if task.plan is not None:
        return _plan_rounds(task, "bank__")
    if task.template == "balance_then_transfer":
        return [0]
    if task.template in _BANK_TEMPLATES:
        return [0]
    return []


def _payment_rounds(task: AgentTask) -> list[int]:
    if task.plan is not None:
        return _plan_rounds(task, "payment__")
    if task.template == "balance_then_transfer":
        return [1]
    if task.template == "transfer_under_cap":
        return [0]
    return []


def candidate_fault_indices(task: AgentTask, attack: AttackSpec, max_agent_steps: int) -> list[int]:
    """Tool-round indices where the attack is likely to affect an on-path tool call."""
    if attack.operator == OperatorId.BENIGN:
        return [0]

    expected = _expected_rounds_for_task(task)
    bank = _bank_rounds(task)
    payment = _payment_rounds(task)

    op = attack.operator
    if op == OperatorId.LLM_WRONG_AMOUNT:
        preferred = payment or expected
    elif op in (OperatorId.MCP_IO_MISMATCH, OperatorId.A2A_TOOLSET_MISMATCH):
        preferred = bank or expected
    elif op == OperatorId.TAINT_PRIOR_OUTPUT:
        preferred = [1] if task.template == "balance_then_transfer" else expected
    else:
        preferred = expected

    indices = [i for i in preferred if i < max_agent_steps]
    if not indices:
        indices = [i for i in expected if i < max_agent_steps]
    if not indices:
        indices = [0]
    return indices


def pick_agent_fault_call_index(
    task: AgentTask,
    attack: AttackSpec,
    rng: random.Random,
    max_agent_steps: int,
) -> int:
    choices = candidate_fault_indices(task, attack, max_agent_steps)
    return rng.choice(choices)
