from __future__ import annotations

import asyncio

from discovery_sim.sim.episode import EpisodeRunner
from discovery_sim.sim.run_config import RunConfig
from discovery_sim.sim.types import OperatorId


def test_multi_step_produces_two_examples_with_prior_calls():
    runner = EpisodeRunner(seed=7, config=RunConfig(mode="scripted", multi_step=True))
    ep = asyncio.run(
        runner.run_one(
            operator_override=OperatorId.BENIGN,
            scenario_index=2,
        )
    )
    assert ep is not None
    assert len(ep.records) == 2
    assert len(ep.examples) == 2
    second = ep.examples[1]
    assert "prior_calls" in second.input_state
    assert len(second.input_state["prior_calls"]) == 1
    assert "ledger_snapshot" in second.input_state
    assert "prior_calls:" in second.input_state["state_text"]


def test_taint_on_second_step():
    runner = EpisodeRunner(seed=7, config=RunConfig(mode="scripted", multi_step=True))
    ep = asyncio.run(
        runner.run_one(
            operator_override=OperatorId.TAINT_PRIOR_OUTPUT,
            scenario_index=2,
        )
    )
    assert ep is not None
    assert ep.operator == OperatorId.TAINT_PRIOR_OUTPUT
    transfer_ex = ep.examples[-1]
    assert transfer_ex.meta.get("taint_map")
    assert transfer_ex.labels.output_correct is False
