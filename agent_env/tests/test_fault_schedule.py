from __future__ import annotations

import random

from discovery_sim.sim.agent_tasks import agent_task_by_id
from discovery_sim.sim.fault_schedule import candidate_fault_indices, pick_agent_fault_call_index
from discovery_sim.sim.operators import operator_catalog
from discovery_sim.sim.types import OperatorId


def test_wrong_amount_targets_payment_round_on_multistep():
    task = agent_task_by_id("agent_balance_then_pay")
    attack = operator_catalog()[OperatorId.LLM_WRONG_AMOUNT]
    assert candidate_fault_indices(task, attack, max_agent_steps=8) == [1]


def test_mcp_io_targets_bank_round():
    task = agent_task_by_id("agent_balance_then_pay")
    attack = operator_catalog()[OperatorId.MCP_IO_MISMATCH]
    assert candidate_fault_indices(task, attack, max_agent_steps=8) == [0]


def test_single_hop_only_index_zero():
    task = agent_task_by_id("agent_balance")
    attack = operator_catalog()[OperatorId.LLM_WRONG_ACCOUNT]
    assert candidate_fault_indices(task, attack, max_agent_steps=8) == [0]


def test_pick_stays_in_candidates():
    rng = random.Random(0)
    task = agent_task_by_id("agent_transfer")
    attack = operator_catalog()[OperatorId.A2A_CALL_INTERCEPTED]
    for _ in range(20):
        idx = pick_agent_fault_call_index(task, attack, rng, max_agent_steps=8)
        assert idx == 0
