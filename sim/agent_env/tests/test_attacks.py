from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from discovery_sim.sim.episode import EpisodeRunner, run_episodes
from discovery_sim.sim.run_config import RunConfig
from discovery_sim.sim.operators import operator_catalog
from discovery_sim.sim.types import OperatorId


@pytest.mark.parametrize(
    "operator,expect_correct,expect_cause",
    [
        (OperatorId.BENIGN, True, "none"),
        (OperatorId.LLM_WRONG_ACCOUNT, False, "agent"),
        (OperatorId.MCP_IO_MISMATCH, False, "adversary"),
        (OperatorId.A2A_CALL_INTERCEPTED, False, "adversary"),
        (OperatorId.MCP_OUTPUT_INJECTION, False, "adversary"),
    ],
)
def test_operator_labels(operator, expect_correct, expect_cause):
    runner = EpisodeRunner(seed=42, config=RunConfig(mode="scripted", multi_step=False))

    async def _run():
        return await runner.run_one(operator_override=operator, scenario_index=0)

    ep = asyncio.run(_run())
    labels = ep.examples[0].labels
    assert labels.output_correct is expect_correct
    assert labels.cause == expect_cause
    assert ep.examples[0].meta["operator"] == operator.value


def test_run_episodes_writes_artifacts(tmp_path: Path):
    summary = asyncio.run(
        run_episodes(RunConfig(episodes=5, seed=1, mode="scripted", multi_step=False), tmp_path)
    )
    assert summary["episodes"] == 5
    assert (tmp_path / "episodes.jsonl").exists()
    assert (tmp_path / "examples.jsonl").exists()
    assert (tmp_path / "laya_train.jsonl").exists()
    assert (tmp_path / "baseline.json").exists()


def test_operator_catalog_covers_hops():
    catalog = operator_catalog()
    hops = {spec.hop for spec in catalog.values() if spec.hop}
    assert hops == {"llm", "a2a", "mcp"}
