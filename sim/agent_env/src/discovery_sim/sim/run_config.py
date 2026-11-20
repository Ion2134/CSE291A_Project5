from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from mcp_hardening import HardeningConfig

from discovery_sim.llm.chat import DEFAULT_TEMPERATURE, ChatBackend


@dataclass
class RunConfig:
    episodes: int = 50
    seed: int = 0
    mode: Literal["agent", "scripted"] = "agent"
    max_agent_steps: int = 8
    llm_samples: int = 0
    llm_backend: ChatBackend = "lmstudio"
    llm_base_url: str | None = None
    llm_model: str | None = None
    multi_step: bool = True
    overwrite_output: bool = False
    concurrency: int = 1
    # Agent-mode knobs (recorded in run_manifest.json). The defaults reproduce earlier runs.
    temperature: float = DEFAULT_TEMPERATURE
    task_set: str = "core"  # One of task_defs.task_set_names(): core (fixed list) or hard (sampled).
    # fixed = original static list; parameterized = paraphrase/entity sampler (agent mode, not for hard).
    task_sample: Literal["fixed", "parameterized"] = "fixed"
    # Restrict / reweight operator sampling. None keeps the full catalog, uniform.
    operators: tuple[str, ...] | None = None
    operator_weights: dict[str, float] = field(default_factory=dict)
    # Agent mode: record every step's internal and external state to <out>/state.db (needs ../../recovery/state_db).
    state_db: bool = False
    # Agent mode: classifiers to ask about every tool call (specs for ../../recovery's recovery_classifiers: a name, or a dict
    # with "name" and its inputs). Empty = none are called. Outputs and the recovery decision are recorded.
    recovery_classifiers: tuple = ()
    # Agent mode: roll a call back (../../recovery) when hardening rejects its response or the classifiers'
    # majority flags it. Needs state_db. At most max_recoveries rollbacks per episode.
    recovery: bool = False
    max_recoveries: int = 2
    # MCP hardening + telemetry toggles (JSON config; default: security off, telemetry on).
    hardening: HardeningConfig = field(default_factory=HardeningConfig)
