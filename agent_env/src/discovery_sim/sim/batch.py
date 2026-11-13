"""``discovery-sim batch``: agent-mode runs over a grid of knobs (model × temperature × task set).

Each grid cell is one run of ``episodes`` episodes with its own seed. With ``max_episodes`` set, only
``max_episodes // episodes`` distinct cells run, picked so that every knob value is covered (see
``pick_cells``). Finished runs are skipped, so a batch can be resumed or extended.

    discovery-sim batch --config configs/batch-example.json --dry-run
"""
from __future__ import annotations

import itertools
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from discovery_sim.sim.episode import run_episodes
from discovery_sim.sim.run_config import RunConfig
from discovery_sim.sim.run_output import RUN_MARKER
from discovery_sim.sim.task_defs import task_set_names
from discovery_sim.sim.task_sample import TASK_SAMPLE_MODES

AGENT_ENV = Path(__file__).resolve().parents[3]   # agent_env/ (batch.py -> sim -> discovery_sim -> src)
KNOBS = ("model", "temperature", "task_set")


@dataclass(frozen=True)
class Cell:
    """One point of the knob grid."""

    model: str
    temperature: float
    task_set: str

    def knob_values(self) -> set[tuple[str, Any]]:
        return {(knob, getattr(self, knob)) for knob in KNOBS}

    def run_name(self, batch: str) -> str:
        model = re.sub(r"[^A-Za-z0-9.]+", "-", self.model).strip("-")
        return f"{batch}-{model}-t{self.temperature:g}-{self.task_set}"


@dataclass(frozen=True)
class BatchConfig:
    """A batch config file (JSON). Paths are relative to ``agent_env/``."""

    name: str
    models: tuple[str, ...]
    temperatures: tuple[float, ...]
    task_sets: tuple[str, ...] = ("core",)
    task_sample: str = "fixed"         # For every core run: fixed or parameterized (hard ignores it).
    episodes: int = 50
    max_episodes: int | None = None    # Episode budget; None runs the whole grid.
    sample_seed: int = 0               # Which cells the budget picks.
    base_seed: int = 0                 # Cell i of the full grid uses seed base_seed + i.
    backend: str = "triton"
    base_url: str | None = None
    concurrency: int = 1
    max_agent_steps: int = 8
    runs_dir: str = "data/runs"
    batches_dir: str = "data/batches"
    extra: dict = field(default_factory=dict, compare=False)

    @classmethod
    def from_file(cls, path: Path) -> "BatchConfig":
        raw = json.loads(Path(path).read_text())
        known = {name for name in cls.__dataclass_fields__ if name != "extra"}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"{path}: unknown batch config keys {sorted(unknown)}")
        for key in ("models", "temperatures", "task_sets"):
            if key in raw:
                raw[key] = tuple(raw[key])
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        if not self.models or not self.temperatures or not self.task_sets:
            raise ValueError("A batch needs at least one model, temperature and task set")
        unknown_sets = set(self.task_sets) - set(task_set_names())
        if unknown_sets:
            raise ValueError(f"Unknown task sets {sorted(unknown_sets)}; expected some of {task_set_names()}")
        if self.task_sample not in TASK_SAMPLE_MODES:
            raise ValueError(f"task_sample must be one of {TASK_SAMPLE_MODES}")
        if self.episodes < 1:
            raise ValueError("episodes must be at least 1")
        if self.max_episodes is not None and self.max_episodes < self.episodes:
            raise ValueError("max_episodes must allow at least one run of `episodes` episodes")

    def grid(self) -> list[Cell]:
        """Every cell, in a fixed order (models, then temperatures, then task sets)."""
        return [Cell(model, float(temperature), task_set)
                for model, temperature, task_set in itertools.product(self.models, self.temperatures, self.task_sets)]


@dataclass(frozen=True)
class PlannedRun:
    cell: Cell
    seed: int
    name: str
    out_dir: Path

    @property
    def done(self) -> bool:
        return (self.out_dir / RUN_MARKER).exists()

    def to_json(self) -> dict:
        return {"run": self.name, "model": self.cell.model, "temperature": self.cell.temperature,
                "task_set": self.cell.task_set, "seed": self.seed}


# ---------------------------------------------------------------- planning

def plan_batch(config: BatchConfig) -> list[PlannedRun]:
    """The runs this batch consists of, in the order they are run."""
    grid = config.grid()
    seeds = {cell: config.base_seed + index for index, cell in enumerate(grid)}
    if config.max_episodes is None:
        cells = grid
    else:
        cells = pick_cells(grid, config.max_episodes // config.episodes, config.sample_seed)

    runs_dir = AGENT_ENV / config.runs_dir
    return [PlannedRun(cell, seeds[cell], cell.run_name(config.name), runs_dir / cell.run_name(config.name))
            for cell in cells]


def pick_cells(cells: list[Cell], count: int, seed: int) -> list[Cell]:
    """``count`` distinct cells that spread over every knob value.

    Cells are picked one at a time: among the remaining cells that add the most knob values not yet
    covered (ties broken by how much the *next* pick could then still add, so a round doesn't paint
    itself into a corner), one is chosen at random. Once every value is covered, coverage starts
    over. The order is deterministic for a seed, so a larger ``count`` keeps the earlier picks.
    """
    rng = random.Random(seed)
    all_values = set().union(*(cell.knob_values() for cell in cells))
    remaining, chosen, covered = list(cells), [], set()
    while remaining and len(chosen) < count:
        scores = [_coverage_score(cell, remaining, covered, all_values) for cell in remaining]
        best = max(scores)
        pick = rng.choice([cell for cell, score in zip(remaining, scores) if score == best])

        chosen.append(pick)
        remaining.remove(pick)
        covered = _cover(covered, pick, all_values)
    return chosen


def _cover(covered: set, cell: Cell, all_values: set) -> set:
    """Coverage after picking ``cell``; it resets once every knob value has been covered."""
    covered = covered | cell.knob_values()
    return set() if covered == all_values else covered


def _coverage_score(cell: Cell, remaining: list[Cell], covered: set, all_values: set) -> tuple[int, int]:
    """(new values this cell covers, the most new values the following pick could cover)."""
    gain = len(cell.knob_values() - covered)
    after = _cover(covered, cell, all_values)
    next_gain = max((len(other.knob_values() - after) for other in remaining if other != cell), default=0)
    return gain, next_gain


# ---------------------------------------------------------------- running

async def run_batch(config: BatchConfig, *, dry_run: bool = False) -> dict:
    """Write the batch file, then run every planned run that hasn't finished yet."""
    planned = plan_batch(config)
    if not dry_run:
        _write_batch_file(config, planned)

    statuses = {}
    for run in planned:
        if run.done:
            statuses[run.name] = "skipped (done)"
        elif dry_run:
            statuses[run.name] = "would run"
        else:
            await run_episodes(_run_config(config, run), run.out_dir)
            statuses[run.name] = "done"
        print(f"{run.name} (seed {run.seed}): {statuses[run.name]}", flush=True)

    return {"batch": config.name, "grid_size": len(config.grid()), "runs": len(planned),
            "episodes": len(planned) * config.episodes, "statuses": statuses}


def _run_config(config: BatchConfig, run: PlannedRun) -> RunConfig:
    return RunConfig(episodes=config.episodes, seed=run.seed, mode="agent", max_agent_steps=config.max_agent_steps,
                     llm_backend=config.backend, llm_base_url=config.base_url, llm_model=run.cell.model,
                     concurrency=max(1, config.concurrency), temperature=run.cell.temperature,
                     task_set=run.cell.task_set, task_sample=config.task_sample)


def _write_batch_file(config: BatchConfig, planned: list[PlannedRun]) -> None:
    """``data/batches/<name>.json``; its ``runs`` list works as a ``--config`` for the MLP/Laya prepare steps."""
    path = AGENT_ENV / config.batches_dir / f"{config.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "name": config.name,
        "runs": [run.name for run in planned],
        "cells": [run.to_json() for run in planned],
        "episodes_per_run": config.episodes,
        "task_sample": config.task_sample,
        "max_episodes": config.max_episodes,
        "sample_seed": config.sample_seed,
        "grid_size": len(config.grid()),
    }, indent=2) + "\n")
