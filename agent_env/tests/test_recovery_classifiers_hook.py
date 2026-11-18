"""Agent runs with ``recovery_classifiers`` ask each one about every tool call and record outputs + the decision."""

import asyncio
import json
import sqlite3

import pytest

from discovery_sim.sim.episode import run_episodes
from discovery_sim.sim.run_config import RunConfig

pytest.importorskip("recovery_classifiers")


def test_run_records_every_classifier_and_the_decision(tmp_path):
    out = tmp_path / "run"
    asyncio.run(run_episodes(RunConfig(episodes=12, seed=5, mode="agent", llm_backend="oracle", task_set="hard",
                                       recovery_classifiers=("oracle", "rules"), state_db=True), out))
    rows = [json.loads(line) for line in (out / "examples.jsonl").read_text().splitlines()]
    assert rows
    for row in rows:
        outputs, decision = row["meta"]["classifier_outputs"], row["meta"]["recovery"]
        assert set(outputs) == {"oracle", "rules"}
        oracle_cause = max(outputs["oracle"], key=outputs["oracle"].get)
        assert oracle_cause == row["label"]["cause"]
        # Two classifiers: a strict majority means both must flag the call.
        both_flag = all(max(p, key=p.get) != "none" for p in outputs.values())
        assert decision["recover"] is both_flag
    assert json.loads((out / "run_manifest.json").read_text())["recovery_classifiers"] == ["oracle", "rules"]

    db = sqlite3.connect(out / "state.db")
    recorded = db.execute("SELECT classifier_outputs, recover FROM tool_calls").fetchall()
    assert len(recorded) == len(rows) and all(json.loads(outputs).keys() == {"oracle", "rules"}
                                              for outputs, _ in recorded)


def test_no_recovery_classifiers_means_nothing_recorded(tmp_path):
    out = tmp_path / "run"
    asyncio.run(run_episodes(RunConfig(episodes=2, seed=5, mode="agent", llm_backend="oracle"), out))
    rows = [json.loads(line) for line in (out / "examples.jsonl").read_text().splitlines()]
    assert all("classifier_outputs" not in row["meta"] and "recovery" not in row["meta"] for row in rows)
    assert "recovery_classifiers" not in json.loads((out / "run_manifest.json").read_text())
