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
