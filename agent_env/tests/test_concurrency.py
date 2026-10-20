from __future__ import annotations

import asyncio

from discovery_sim.sim.episode import run_episodes
from discovery_sim.sim.run_config import RunConfig


def test_concurrent_scripted_run(tmp_path):
    out = tmp_path / "conc"
    summary = asyncio.run(
        run_episodes(
            RunConfig(episodes=12, seed=99, mode="scripted", concurrency=4),
            out,
        )
    )
    assert summary["episodes"] == 12
    assert summary["concurrency"] == 4
    assert (out / "episodes.jsonl").read_text().count("\n") == 12
