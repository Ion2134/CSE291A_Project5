from __future__ import annotations

import random

import pytest

from discovery_sim.sim.operators import (
    apply_llm_attack,
    parse_operator_ids,
    parse_operator_weights,
    pick_operator,
)
from discovery_sim.sim.types import BaseTask, OperatorId, ToolCall


def test_parse_operator_ids_and_weights():
    assert parse_operator_ids("llm_wrong_account, benign") == [
        OperatorId.LLM_WRONG_ACCOUNT,
        OperatorId.BENIGN,
    ]
    weights = parse_operator_weights("llm_wrong_account=4,benign=1")
    assert weights[OperatorId.LLM_WRONG_ACCOUNT] == 4.0
    assert weights[OperatorId.BENIGN] == 1.0


def test_parse_operator_rejects_unknown():
    with pytest.raises(ValueError, match="unknown operator"):
        parse_operator_ids("not_an_op")
    with pytest.raises(ValueError, match="unknown operator"):
        parse_operator_weights("nope=1")


def test_weighted_pick_prefers_heavy_operators():
    rng = random.Random(0)
    counts = {OperatorId.LLM_WRONG_ACCOUNT: 0, OperatorId.BENIGN: 0}
    for _ in range(400):
        spec = pick_operator(
            rng,
            allowed=[OperatorId.LLM_WRONG_ACCOUNT, OperatorId.BENIGN],
            weights={OperatorId.LLM_WRONG_ACCOUNT: 9, OperatorId.BENIGN: 1},
        )
        counts[spec.operator] += 1
    assert counts[OperatorId.LLM_WRONG_ACCOUNT] > counts[OperatorId.BENIGN] * 3


def test_allowed_operators_restrict_catalog():
    rng = random.Random(1)
    for _ in range(50):
        spec = pick_operator(rng, allowed=[OperatorId.LLM_WRONG_AMOUNT])
        assert spec.operator == OperatorId.LLM_WRONG_AMOUNT


def _balance_task(account: str = "acct_synth_0031") -> BaseTask:
    return BaseTask(
        task_id="t",
        template="balance_lookup",
        user_message=f"balance {account}",
        mandate={"account_id": account},
        target_agent="bank",
        tool_name="get_balance",
        correct_arguments={"account_id": account},
    )


def test_wrong_account_differs_from_mandate():
    task = _balance_task("acct_synth_0031")
    call = ToolCall("get_balance", {"account_id": "acct_synth_0031"}, "bank")
    rng = random.Random(7)
    accounts = ["acct_synth_0031", "acct_synth_0042", "acct_synth_0055", "acct_synth_0068"]
    seen = set()
    for _ in range(20):
        mutated = apply_llm_attack(
            task, call, OperatorId.LLM_WRONG_ACCOUNT, rng=rng, account_ids=accounts
        )
        assert mutated.arguments["account_id"] != "acct_synth_0031"
        seen.add(mutated.arguments["account_id"])
    assert len(seen) > 1


def test_wrong_amount_is_diversified():
    task = BaseTask(
        task_id="pay",
        template="transfer_under_cap",
        user_message="pay",
        mandate={"from_account_id": "acct_synth_0031", "payee_id": "payee_synth_0007", "amount_cents": 5000},
        target_agent="payment",
        tool_name="initiate_transfer",
        correct_arguments={
            "from_account_id": "acct_synth_0031",
            "payee_id": "payee_synth_0007",
            "amount_cents": 5000,
        },
    )
    call = ToolCall(
        "initiate_transfer",
        {"from_account_id": "acct_synth_0031", "payee_id": "payee_synth_0007", "amount_cents": 5000},
        "payment",
    )
    rng = random.Random(3)
    amounts = {
        apply_llm_attack(task, call, OperatorId.LLM_WRONG_AMOUNT, rng=rng).arguments["amount_cents"]
        for _ in range(30)
    }
    assert 5000 not in amounts
    assert len(amounts) > 1
