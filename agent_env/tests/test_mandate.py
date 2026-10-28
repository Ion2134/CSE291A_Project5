from __future__ import annotations

import asyncio

from discovery_sim.llm.chat import ChatTurn, ScriptChatClient, ToolCallRequest
from discovery_sim.sim.agent_tasks import agent_task_index
from discovery_sim.sim.episode import EpisodeRunner
from discovery_sim.sim.mandate import expected_arguments_for_tool
from discovery_sim.sim.run_config import RunConfig
from discovery_sim.sim.types import OperatorId

TRANSFER_MANDATE = {
    "from_account_id": "acct_synth_0031",
    "payee_id": "payee_synth_0007",
    "amount_cents": 2500,
    "cap_cents": 100_000,
}


def _run(task_id: str, calls: list[tuple[str, dict]]):
    turns = [ChatTurn(content=None, tool_calls=[ToolCallRequest(f"c{i}", n, a)]) for i, (n, a) in enumerate(calls)]
    turns.append(ChatTurn(content="Done.", tool_calls=[]))
    runner = EpisodeRunner(seed=1, config=RunConfig(mode="agent"), chat_client=ScriptChatClient(turns))
    return asyncio.run(
        runner.run_one(operator_override=OperatorId.BENIGN, scenario_index=agent_task_index(task_id))
    )


def test_get_balance_on_transfer_mandate_uses_from_account():
    assert expected_arguments_for_tool("get_balance", TRANSFER_MANDATE) == {"account_id": "acct_synth_0031"}


def test_expected_arguments_never_raise_on_missing_keys():
    assert expected_arguments_for_tool("get_balance", {"action": "list_accounts"}) == {"account_id": None}
    assert expected_arguments_for_tool("initiate_transfer", {"account_id": "acct_synth_0031"}) == {
        "from_account_id": None,
        "payee_id": None,
        "amount_cents": None,
    }


def test_balance_check_before_transfer_is_labeled_correct():
    """Regression for data/runs/triton42: this sequence used to crash with KeyError: 'account_id'."""
    ep = _run(
        "agent_transfer_small",
        [
            ("bank__get_balance", {"account_id": "acct_synth_0031"}),
            (
                "payment__initiate_transfer",
                {"from_account_id": "acct_synth_0031", "payee_id": "payee_synth_0007", "amount_cents": 2500},
            ),
        ],
    )
    assert ep is not None
    assert len(ep.records) == 2
    assert [ex.labels.output_correct for ex in ep.examples] == [True, True]
    assert ep.task_correct


def test_balance_check_on_wrong_account_is_labeled_wrong():
    ep = _run("agent_transfer_small", [("bank__get_balance", {"account_id": "acct_synth_0042"})])
    assert ep is not None
    assert ep.examples[0].labels.output_correct is False


def test_off_mandate_calls_are_labeled_wrong_not_crashed():
    # A balance lookup on a task that names no account, and a transfer on a read-only task.
    ep = _run("agent_list_accounts", [("bank__get_balance", {"account_id": "acct_synth_0031"})])
    assert ep is not None and ep.examples[0].labels.output_correct is False

    ep = _run(
        "agent_balance",
        [
            (
                "payment__initiate_transfer",
                {"from_account_id": "acct_synth_0031", "payee_id": "payee_synth_0007", "amount_cents": 2500},
            )
        ],
    )
    assert ep is not None and ep.examples[0].labels.output_correct is False
