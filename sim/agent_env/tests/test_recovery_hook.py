"""Recovery (../../recovery): a flagged call is rolled back to the step before it and the agent carries on."""

import asyncio
import json
import sqlite3

import pytest

from discovery_sim.ledger.store import SyntheticLedger
from discovery_sim.llm.chat import ChatTurn, ScriptChatClient, ToolCallRequest
from discovery_sim.sim.agent_tasks import agent_task_index, agent_tasks
from discovery_sim.sim.episode import run_episodes
from discovery_sim.sim.harness import AgentPool
from discovery_sim.sim.operators import operator_catalog
from discovery_sim.sim.orchestrator import AgentOrchestrator
from discovery_sim.sim.run_config import RunConfig
from discovery_sim.sim.types import OperatorId

recovery = pytest.importorskip("recovery")
state_db = pytest.importorskip("state_db")

TRANSFER = {"from_account_id": "acct_synth_0031", "payee_id": "payee_synth_0007", "amount_cents": 5000}


def test_restore_refills_the_same_ledger():
    ledger = SyntheticLedger()
    before = ledger.snapshot()
    ledger.initiate_transfer("acct_synth_0031", "payee_synth_0007", 5000)
    same_object = ledger.accounts
    ledger.restore(before)
    assert ledger.snapshot() == before and ledger.payments == {}
    assert ledger.accounts is not same_object  # refilled, but the ledger object itself is the same one


def run_transfer(tmp_path, operator, *, hardening=None, panel=None, max_recoveries=2):
    """One agent episode: a transfer (attacked by ``operator``), then "Done."; returns (result, pool, store)."""
    from mcp_hardening import HardeningRuntime, Keyring, load_config

    chat = ScriptChatClient([ChatTurn(content=None, tool_calls=[ToolCallRequest("c1", "payment__initiate_transfer",
                                                                                 dict(TRANSFER))]),
                             ChatTurn(content="Done.", tool_calls=[])])
    runtime = HardeningRuntime.build(load_config(hardening), Keyring.ephemeral_key()) if hardening else None
    pool = AgentPool(SyntheticLedger(), runtime=runtime)
    store = state_db.StateStore(tmp_path / "state.db")
    task = agent_tasks("core")[agent_task_index("agent_transfer")]
    result = asyncio.run(AgentOrchestrator(chat, store=store, panel=panel, recovery=True,
                                           max_recoveries=max_recoveries).run_episode(
        task, pool, operator_catalog()[operator], 0, seed=1, episode_id="ep_test"))
    return result, pool, store


def balance(pool):
    return pool.ledger.get_balance("acct_synth_0031")["balance_cents"]


def test_classifier_flag_rolls_the_transfer_back(tmp_path):
    from recovery_classifiers import Panel
    start = balance(AgentPool(SyntheticLedger()))
    result, pool, store = run_transfer(tmp_path, OperatorId.LLM_WRONG_AMOUNT, panel=Panel(["oracle"]))
    meta = result.examples[0].meta["recovery"]
    assert meta["recover"] and meta["rolled_back"] and meta["reason"] == "classifiers"
    assert pool.ledger.payments == {} and balance(pool) == start  # the wrong transfer was undone
    assert store.tool_calls_at("ep_test", 10**9) == []             # and is gone from the current state
    messages = store.messages_at("ep_test", 10**9)
    assert json.loads(messages[-2]["content"]) == recovery.BLOCKED_REPLY  # the agent was told...
    assert messages[-1] == {"role": "assistant", "content": "Done."}     # ...and carried on
    assert result.task_completed


def test_response_signature_failure_rolls_back_without_classifiers(tmp_path):
    start = balance(AgentPool(SyntheticLedger()))
    result, pool, _ = run_transfer(tmp_path, OperatorId.MCP_OUTPUT_INJECTION, hardening="all_on")
    record, meta = result.records[0], result.examples[0].meta["recovery"]
    assert record.hardening["failure_stage"] == "response"
    assert meta["rolled_back"] and meta["reason"] == "response_sig_failed"
    assert pool.ledger.payments == {} and balance(pool) == start  # the transfer had gone through: undone


def test_request_rejection_is_not_rolled_back(tmp_path):
    from recovery_classifiers import Panel
    result, pool, store = run_transfer(tmp_path, OperatorId.A2A_CALL_INTERCEPTED, hardening="all_on",
                                       panel=Panel(["oracle"]))
    meta = result.examples[0].meta
    assert result.records[0].hardening["failure_stage"] == "request"
    assert meta["classifier_outputs"] is None  # the classifiers are not asked about blocked calls
    assert meta["recovery"]["reason"] == "blocked_before_tool" and not meta["recovery"]["rolled_back"]
    assert pool.ledger.payments == {}          # never ran in the first place
    assert len(store.tool_calls_at("ep_test", 10**9)) == 1  # nothing was deleted


def test_cap_of_zero_never_rolls_back(tmp_path):
    from recovery_classifiers import Panel
    result, pool, _ = run_transfer(tmp_path, OperatorId.LLM_WRONG_AMOUNT, panel=Panel(["oracle"]), max_recoveries=0)
    assert result.examples[0].meta["recovery"]["reason"] == "cap_reached"
    assert len(pool.ledger.payments) == 1


def test_recovery_needs_the_state_db(tmp_path):
    with pytest.raises(ValueError, match="state"):
        asyncio.run(run_episodes(RunConfig(episodes=1, mode="agent", llm_backend="oracle", recovery=True), tmp_path))


def test_run_with_recovery_keeps_files_and_db_consistent(tmp_path):
    out = tmp_path / "run"
    asyncio.run(run_episodes(RunConfig(episodes=30, seed=4, mode="agent", llm_backend="oracle", task_set="hard",
                                       state_db=True, recovery=True, recovery_classifiers=("oracle",)), out))
    rows = [json.loads(line) for line in (out / "examples.jsonl").read_text().splitlines()]
    rolled = [r for r in rows if r["meta"]["recovery"]["rolled_back"]]
    assert rolled and all(r["label"]["cause"] != "none" for r in rolled)  # oracle flags exactly the errors
    db = sqlite3.connect(out / "state.db")
    # Rolled-back calls stay in examples.jsonl (training data) but are gone from the current state.
    assert db.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0] == len(rows) - len(rolled)
    assert db.execute("SELECT COUNT(*) FROM tool_calls WHERE call_id IS NULL").fetchone()[0] == 0
    baseline = json.loads((out / "baseline.json").read_text())
    assert baseline["recoveries"]["total"] == len(rolled)
    assert json.loads((out / "run_manifest.json").read_text())["recovery"] is True
