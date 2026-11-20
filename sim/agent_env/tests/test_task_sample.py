from __future__ import annotations

import random

from discovery_sim.sim.task_sample import choose_agent_task, paraphrase_request_keys, sample_agent_task
from discovery_sim.sim.types import OperatorId


def test_paraphrases_yield_distinct_request_groups():
    keys = paraphrase_request_keys()
    assert len(keys) == len(set(keys))
    assert len(keys) >= 20


def test_parameterized_tasks_fill_mandates():
    rng = random.Random(11)
    for _ in range(40):
        task = sample_agent_task(rng)
        assert task.user_message
        assert task.template
        if task.template == "list_accounts":
            assert task.mandate.get("action") == "list_accounts"
        elif task.template == "balance_lookup":
            assert task.mandate["account_id"].startswith("acct_synth_")
            assert task.mandate["account_id"] in task.user_message
        elif task.template in ("transfer_under_cap", "balance_then_transfer"):
            assert task.mandate["from_account_id"].startswith("acct_synth_")
            assert task.mandate["payee_id"].startswith("payee_synth_")
            assert int(task.mandate["amount_cents"]) > 0


def test_prefer_payment_when_amount_operator_present():
    rng = random.Random(0)
    paymentish = 0
    for _ in range(80):
        task = choose_agent_task(
            rng,
            task_sample="parameterized",
            operators=[OperatorId.LLM_WRONG_AMOUNT, OperatorId.BENIGN],
        )
        if task.template in ("transfer_under_cap", "balance_then_transfer"):
            paymentish += 1
    assert paymentish > 40


def test_fixed_sample_keeps_core_list():
    rng = random.Random(2)
    task = choose_agent_task(rng, task_sample="fixed", scenario_index=0)
    assert task.task_id == "agent_balance"
