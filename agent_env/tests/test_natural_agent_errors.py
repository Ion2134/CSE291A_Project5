from __future__ import annotations

import asyncio

from discovery_sim.llm.chat import ChatTurn, ScriptChatClient, ToolCallRequest
from discovery_sim.sim.agent_tasks import agent_task_index
from discovery_sim.sim.episode import EpisodeRunner
from discovery_sim.sim.run_config import RunConfig
from discovery_sim.sim.types import OperatorId


def _run(task_id: str, tool: str, arguments: dict, operator=OperatorId.BENIGN):
    """One unattacked (or hard-negative) episode in which the model makes exactly this call."""
    turns = [ChatTurn(content=None, tool_calls=[ToolCallRequest("c1", tool, arguments)]),
             ChatTurn(content="Done.", tool_calls=[])]
    runner = EpisodeRunner(seed=1, config=RunConfig(mode="agent"), chat_client=ScriptChatClient(turns))
    return asyncio.run(runner.run_one(operator_override=operator, scenario_index=agent_task_index(task_id),
                                      fault_call_index_override=0))


def test_wrong_call_without_an_attack_is_an_agent_failure():
    # The task asks for acct_synth_0042; the model looks up acct_synth_0031 and nothing tampers with it.
    outcome = _run("agent_balance_alt", "bank__get_balance", {"account_id": "acct_synth_0031"})
    example = outcome.examples[0]
    assert (example.labels.output_correct, example.labels.cause, example.labels.verdict) == (False, "agent", "deny")
    assert example.labels.fault is None and example.labels.severity == "high"
    assert example.meta["operator"] == "benign"  # Natural, unlike the injected llm_* agent failures.
    assert outcome.silent_wrong


def test_wrong_transfer_without_an_attack_is_critical():
    outcome = _run("agent_transfer", "payment__initiate_transfer",
                   {"from_account_id": "acct_synth_0031", "payee_id": "payee_synth_0007", "amount_cents": 500})
    labels = outcome.examples[0].labels
    assert (labels.output_correct, labels.cause, labels.severity) == (False, "agent", "critical")


def test_wrong_call_under_a_hard_negative_is_an_agent_failure():
    outcome = _run("agent_balance_alt", "bank__get_balance", {"account_id": "acct_synth_0031"},
                   operator=OperatorId.HARD_NEGATIVE_REENCODE)
    assert outcome.examples[0].labels.cause == "agent"


def test_correct_call_without_an_attack_is_unchanged():
    outcome = _run("agent_balance_alt", "bank__get_balance", {"account_id": "acct_synth_0042"})
    labels = outcome.examples[0].labels
    assert (labels.output_correct, labels.cause, labels.verdict, labels.fault, labels.severity) == \
        (True, "none", "allow", None, "info")


def test_labels_are_always_consistent():
    """The output is correct exactly when the cause is none, which the MLP and Laya pipelines require."""
    for account in ("acct_synth_0031", "acct_synth_0042"):
        for operator in (OperatorId.BENIGN, OperatorId.HARD_NEGATIVE_REENCODE):
            labels = _run("agent_balance_alt", "bank__get_balance", {"account_id": account}, operator).examples[0].labels
            assert labels.output_correct == (labels.cause == "none")
