from __future__ import annotations

import asyncio
import json
import random

import pytest

from discovery_sim.ledger.store import SyntheticLedger
from discovery_sim.llm.chat import ChatTurn, OracleChat, ScriptChatClient, ToolCallRequest
from discovery_sim.sim.episode import run_episodes
from discovery_sim.sim.fault_schedule import candidate_fault_indices
from discovery_sim.sim.harness import AgentPool
from discovery_sim.sim.hard_tasks import HARD_FAMILIES, hard_phrases
from discovery_sim.sim.operators import operator_catalog
from discovery_sim.sim.orchestrator import AgentOrchestrator
from discovery_sim.sim.run_config import RunConfig
from discovery_sim.sim.task_sample import choose_agent_task, paraphrase_request_keys
from discovery_sim.sim.types import OperatorId

CATALOG = operator_catalog()


def _family_task(family: str, where=lambda task: True, **wanted):
    """The first ``family`` task whose mandate has every ``wanted`` key/value and that passes ``where``."""
    rng = random.Random(0)
    for _ in range(500):
        task = HARD_FAMILIES[family](rng, SyntheticLedger())
        if all(task.mandate.get(key) == value for key, value in wanted.items()) and where(task):
            return task
    raise AssertionError(f"no matching {family} task")


def _run(task, chat, operator=OperatorId.BENIGN, fault_call_index=0):
    orchestrator = AgentOrchestrator(chat, max_steps=8)
    return asyncio.run(orchestrator.run_episode(task, AgentPool(SyntheticLedger()), CATALOG[operator],
                                                fault_call_index, seed=1, rng=random.Random(1)))


def _call(name, arguments, call_id="c1"):
    return ChatTurn(content=None, tool_calls=[ToolCallRequest(call_id, name, arguments)])


# ---------------------------------------------------------------- what the agent and monitor see

@pytest.mark.parametrize("family", sorted(HARD_FAMILIES))
def test_the_answer_is_not_shown(family):
    rng = random.Random(3)
    for _ in range(30):
        task = HARD_FAMILIES[family](rng, SyntheticLedger())
        visible = task.instruction
        assert visible is not None and visible != task.mandate
        assert "read_accounts" not in visible and "no_transfer" not in visible
        if family in ("named_account_transfer", "compare_then_transfer"):
            assert "from_account_id" not in visible  # The source account must be worked out.
        if family == "derived_transfer":
            assert "amount_cents" not in visible  # The amount must be worked out.
        if family == "named_payee_transfer":
            assert "payee_id" not in visible and task.mandate["payee_id"] not in task.user_message


def test_trusted_line_and_prompt_show_only_the_instruction():
    task = _family_task("derived_transfer")
    result = _run(task, OracleChat())
    prompt = result.messages[1]["content"]
    assert json.dumps(task.instruction) in prompt and str(task.mandate["amount_cents"]) not in prompt
    assert f"trusted: {task.instruction}" in result.examples[0].input_state["state_text"]


def test_hard_phrases_are_distinct_from_each_other_and_the_simple_ones():
    assert len(set(hard_phrases())) == len(hard_phrases())
    assert len(paraphrase_request_keys()) == len(set(paraphrase_request_keys()))


# ---------------------------------------------------------------- labels

@pytest.mark.parametrize("family", sorted(HARD_FAMILIES))
def test_oracle_solves_every_family(family):
    rng = random.Random(5)
    for _ in range(15):
        task = HARD_FAMILIES[family](rng, SyntheticLedger())
        result = _run(task, OracleChat())
        assert result.task_correct, task.user_message
        assert all(ex.labels.cause == "none" and ex.labels.output_correct for ex in result.examples)
        assert len(result.examples) == len(task.plan)


def test_conditional_false_branch_paying_is_an_agent_error():
    task = _family_task("conditional_transfer", no_transfer=True)
    account = task.mandate["account_id"]
    pay = {"from_account_id": account, "payee_id": task.instruction["payee_id"],
           "amount_cents": task.instruction["amount_cents"]}
    result = _run(task, ScriptChatClient([_call("bank__get_balance", {"account_id": account}),
                                          _call("payment__initiate_transfer", pay, "c2")]))
    read, transfer = result.examples
    assert read.labels.cause == "none"
    assert (transfer.labels.output_correct, transfer.labels.cause) == (False, "agent")
    assert not result.task_correct


def test_reading_an_allowed_account_is_fine_but_others_are_not():
    task = _family_task("compare_then_transfer", where=lambda t: len(t.mandate["read_accounts"]) == 2)
    first, second = task.mandate["read_accounts"]
    outsider = next(a for a in SyntheticLedger().accounts if a not in (first, second))
    result = _run(task, ScriptChatClient([_call("bank__get_balance", {"account_id": second}),
                                          _call("bank__get_balance", {"account_id": outsider}, "c2")]))
    allowed, other = result.examples
    assert allowed.labels.cause == "none" and other.labels.cause == "agent"


class _HalfOfWhatISee:
    """A faithful agent for a 'pay half the balance' task: it pays half of the balance it observes."""

    model_id = "reactive"

    def __init__(self, task):
        self.task = task

    async def complete(self, messages, tools):
        tool_outputs = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
        account, payee = self.task.mandate["from_account_id"], self.task.mandate["payee_id"]
        if not tool_outputs:
            return _call("bank__get_balance", {"account_id": account})
        if len(tool_outputs) == 1:
            seen = tool_outputs[0]["balance_cents"]
            return _call("payment__initiate_transfer",
                         {"from_account_id": account, "payee_id": payee, "amount_cents": seen // 2}, "c2")
        return ChatTurn(content="Done.")


def _half_task():
    # Not acct_synth_0042: the IO-mismatch attack swaps in that account's balance, so it must differ.
    return _family_task("derived_transfer", where=lambda t: t.instruction["amount"].startswith("half")
                        and t.mandate["from_account_id"] != "acct_synth_0042")


def test_tampered_read_makes_the_downstream_transfer_a_propagated_adversary_error():
    task = _half_task()
    result = _run(task, _HalfOfWhatISee(task), operator=OperatorId.MCP_IO_MISMATCH, fault_call_index=0)
    read, transfer = result.examples
    assert read.labels.cause == "adversary" and not read.labels.propagated
    assert (transfer.labels.output_correct, transfer.labels.cause) == (False, "adversary")
    assert transfer.labels.propagated and transfer.labels.propagated_from == 0


def test_same_agent_without_tampering_is_correct():
    task = _half_task()
    result = _run(task, _HalfOfWhatISee(task))
    assert result.task_correct and not any(ex.labels.propagated for ex in result.examples)


def test_wrong_transfer_without_earlier_tampering_stays_an_agent_error():
    task = _half_task()
    wrong = {"from_account_id": task.mandate["from_account_id"], "payee_id": task.mandate["payee_id"],
             "amount_cents": task.mandate["amount_cents"] + 100}
    result = _run(task, ScriptChatClient([_call("bank__get_balance", {"account_id": wrong["from_account_id"]}),
                                          _call("payment__initiate_transfer", wrong, "c2")]))
    transfer = result.examples[1]
    assert transfer.labels.cause == "agent" and not transfer.labels.propagated


class _PayWhoeverHasTheName:
    """A faithful agent for a named-payee task: it pays the payee id that list_payees gives for the name."""

    model_id = "reactive"

    def __init__(self, task, payee_name):
        self.task, self.payee_name = task, payee_name

    async def complete(self, messages, tools):
        tool_outputs = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
        if not tool_outputs:
            return _call("bank__list_payees", {})
        if len(tool_outputs) == 1:
            payee = next(p["payee_id"] for p in tool_outputs[0]["payees"] if p["name"] == self.payee_name)
            return _call("payment__initiate_transfer", {"from_account_id": self.task.mandate["from_account_id"],
                                                        "payee_id": payee,
                                                        "amount_cents": self.task.mandate["amount_cents"]}, "c2")
        return ChatTurn(content="Done.")


def _payee_task():
    task = _family_task("named_payee_transfer")
    return task, SyntheticLedger().payees[task.mandate["payee_id"]].name


def test_payee_lookup_without_tampering_is_correct():
    task, name = _payee_task()
    result = _run(task, _PayWhoeverHasTheName(task, name))
    assert result.task_correct and all(ex.labels.cause == "none" for ex in result.examples)


def test_tampered_payee_list_makes_the_transfer_a_propagated_adversary_error():
    task, name = _payee_task()
    result = _run(task, _PayWhoeverHasTheName(task, name), operator=OperatorId.MCP_IO_MISMATCH, fault_call_index=0)
    lookup, transfer = result.examples
    assert (lookup.labels.output_correct, lookup.labels.cause) == (False, "adversary")
    assert (transfer.labels.cause, transfer.labels.propagated_from) == ("adversary", 0)
    assert transfer.input_state["model_call"]["payee_id"] != task.mandate["payee_id"]


def test_paying_the_wrong_payee_on_its_own_is_an_agent_error():
    task, _ = _payee_task()
    wrong = next(p for p in SyntheticLedger().payees if p != task.mandate["payee_id"])
    pay = {"from_account_id": task.mandate["from_account_id"], "payee_id": wrong,
           "amount_cents": task.mandate["amount_cents"]}
    result = _run(task, ScriptChatClient([_call("bank__list_payees", {}), _call("payment__initiate_transfer", pay, "c2")]))
    assert result.examples[1].labels.cause == "agent" and not result.examples[1].labels.propagated


# ---------------------------------------------------------------- scheduling and sampling

def test_attacks_land_on_the_planned_rounds():
    task = _family_task("derived_transfer")
    assert candidate_fault_indices(task, CATALOG[OperatorId.LLM_WRONG_AMOUNT], 8) == [1]
    assert candidate_fault_indices(task, CATALOG[OperatorId.MCP_IO_MISMATCH], 8) == [0]
    no_pay = _family_task("conditional_transfer", no_transfer=True)
    assert candidate_fault_indices(no_pay, CATALOG[OperatorId.LLM_WRONG_AMOUNT], 8) == [0]


def test_hard_is_a_task_set():
    rng = random.Random(4)
    hard = {choose_agent_task(rng, task_set="hard").template for _ in range(60)}
    assert hard == set(HARD_FAMILIES)
    # The wording knob only applies to fixed task sets; hard tasks are always sampled.
    assert {choose_agent_task(rng, task_set="hard", task_sample="parameterized").template
            for _ in range(60)} <= set(HARD_FAMILIES)
    with pytest.raises(ValueError, match="unknown task set"):
        choose_agent_task(rng, task_set="nope")
    with pytest.raises(ValueError, match="unknown task sample"):
        choose_agent_task(rng, task_sample="nope")


def test_hard_run_end_to_end_writes_consistent_labels(tmp_path):
    out = tmp_path / "hard"
    asyncio.run(run_episodes(RunConfig(episodes=40, seed=8, mode="agent", llm_backend="oracle", task_set="hard"),
                             out))
    rows = [json.loads(line) for line in (out / "examples.jsonl").read_text().splitlines()]
    assert rows and all(r["label"]["output_correct"] == (r["label"]["cause"] == "none") for r in rows)
    assert {r["meta"]["template"] for r in rows} <= set(HARD_FAMILIES)
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["task_set"] == "hard" and manifest["task_count"] is None
