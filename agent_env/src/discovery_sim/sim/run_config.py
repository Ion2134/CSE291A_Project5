from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

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
