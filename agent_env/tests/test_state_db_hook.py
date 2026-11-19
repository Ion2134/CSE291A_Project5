"""Agent runs with ``state_db=True`` record every step to <out>/state.db (the ../state_db package)."""

import asyncio
import json
import sqlite3

import pytest

from discovery_sim.ledger.store import SyntheticLedger
from discovery_sim.sim.episode import run_episodes
from discovery_sim.sim.run_config import RunConfig

state_db = pytest.importorskip("state_db")


def test_snapshot_has_every_ledger_collection():
    ledger = SyntheticLedger()
    before = ledger.snapshot()
    ledger.initiate_transfer("acct_synth_0031", "payee_synth_0007", 1000)
    after = ledger.snapshot()
    assert {len(before[k]) for k in ("accounts", "payees")} == {8, 4} and before["payments"] == []
    assert len(after["payments"]) == 1 and len(after["transactions"]) == len(before["transactions"]) + 1


def test_agent_run_records_every_step(tmp_path):
    out = tmp_path / "run"
    asyncio.run(run_episodes(RunConfig(episodes=6, seed=3, mode="agent", llm_backend="oracle", task_set="hard",
                                       state_db=True), out))
    db = sqlite3.connect(out / "state.db")
    episodes = [json.loads(line)["episode_id"] for line in (out / "episodes.jsonl").read_text().splitlines()]
    examples = (out / "examples.jsonl").read_text().splitlines()

    starts = [row[0] for row in db.execute("SELECT episode_id FROM steps WHERE kind = 'start'")]
    assert sorted(starts) == sorted(episodes)  # the DB uses the same episode ids as the run files
    assert db.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0] == len(examples)
    assert json.loads((out / "run_manifest.json").read_text())["state_db"] == "state.db"
    assert (out / "state.log").read_text().count("\n") == db.execute("SELECT COUNT(*) FROM steps").fetchone()[0]

    # After a completed transfer, that step's ledger snapshot has the payment and the lower balance.
    store = state_db.StateStore(out / "state.db")
    seq, episode_id, call = next((seq, ep, json.loads(output)) for seq, ep, output in db.execute(
        "SELECT seq, episode_id, output FROM tool_calls WHERE tool = 'initiate_transfer' AND cause = 'none'"))
    ledger = store.ledger_at(episode_id, seq)
    assert call["payment_id"] in {p["payment_id"] for p in ledger["payments"]}
    start_seq = next(s for (s,) in db.execute("SELECT seq FROM steps WHERE kind = 'start' AND episode_id = ?",
                                              (episode_id,)))
    balance = {a["account_id"]: a["balance_cents"] for a in store.ledger_at(episode_id, start_seq)["accounts"]}
    after = {a["account_id"]: a["balance_cents"] for a in ledger["accounts"]}
    assert after[call["from_account_id"]] == balance[call["from_account_id"]] - call["amount_cents"]
    # The conversation is there in order, starting with the system prompt.
    assert store.messages_at(episode_id, seq)[0]["role"] == "system"
    store.close()


def test_runs_without_the_flag_write_no_state_files(tmp_path):
    out = tmp_path / "run"
    asyncio.run(run_episodes(RunConfig(episodes=2, seed=3, mode="agent", llm_backend="oracle"), out))
    assert not (out / "state.db").exists() and not (out / "state.log").exists()
    assert "state_db" not in json.loads((out / "run_manifest.json").read_text())


class _RecordingStore:
    """Stand-in for state_db.StateStore that remembers which episode id every write used."""

    def __init__(self):
        self.ids = []

    def start_episode(self, episode_id, ledger):
        self.ids.append(episode_id)

    def record_message(self, episode_id, position, message):
        self.ids.append(episode_id)

    def record_tool_call(self, episode_id, call, ledger):
        self.ids.append(episode_id)


def test_orchestrator_records_under_the_given_episode_id():
    """run_episode(episode_id=...) must hand that id to the state recorder before the loop starts,
    not stamp it on the result afterwards (the merge of the hardening and state DB branches did the latter)."""
    from discovery_sim.llm.chat import ChatTurn, ScriptChatClient, ToolCallRequest
    from discovery_sim.sim.agent_tasks import agent_task_index, agent_tasks
    from discovery_sim.sim.harness import AgentPool
    from discovery_sim.sim.operators import operator_catalog
    from discovery_sim.sim.orchestrator import AgentOrchestrator
    from discovery_sim.sim.types import OperatorId

    chat = ScriptChatClient([
        ChatTurn(content=None, tool_calls=[ToolCallRequest("c1", "bank__get_balance", {"account_id": "acct_synth_0031"})]),
        ChatTurn(content="Done.", tool_calls=[]),
    ])
    store = _RecordingStore()
    task = agent_tasks("core")[agent_task_index("agent_balance_then_pay")]
    result = asyncio.run(AgentOrchestrator(chat, store=store).run_episode(
        task, AgentPool(SyntheticLedger()), operator_catalog()[OperatorId.BENIGN], 0, seed=1, episode_id="ep_fixed01"))

    assert result.episode_id == "ep_fixed01"
    assert store.ids and set(store.ids) == {"ep_fixed01"}


def test_state_db_and_hardening_share_episode_ids(tmp_path):
    """With the state DB, all hardening checks and telemetry on together, every output keys episodes the same way."""
    from mcp_hardening import load_config, read_telemetry, semconv

    out = tmp_path / "run"
    asyncio.run(run_episodes(RunConfig(episodes=20, seed=7, mode="agent", llm_backend="oracle", state_db=True,
                                       hardening=load_config("all_on")), out))
    episodes = {json.loads(line)["episode_id"] for line in (out / "episodes.jsonl").read_text().splitlines()}
    db = sqlite3.connect(out / "state.db")

    assert {r[0] for r in db.execute("SELECT episode_id FROM steps WHERE kind = 'start'")} == episodes
    assert {r[0] for r in db.execute("SELECT DISTINCT episode_id FROM tool_calls")} <= episodes
    rows = read_telemetry(out / "telemetry").to_pylist()
    agent_ids = {r[semconv.GEN_AI_CONVERSATION_ID] for r in rows if r[semconv.GEN_AI_OPERATION_NAME] == "invoke_agent"}
    assert agent_ids == episodes
    # Blocked calls still reach the DB (as failed calls), so it has one row per labelled example.
    baseline = json.loads((out / "baseline.json").read_text())
    assert baseline["hardening_blocked_calls"]["total"] > 0
    assert db.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0] == len(
        (out / "examples.jsonl").read_text().splitlines())
