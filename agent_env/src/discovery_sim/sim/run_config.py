from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class RunConfig:
    episodes: int = 50
    seed: int = 0
    mode: Literal["agent", "scripted"] = "agent"
    max_agent_steps: int = 8
    llm_samples: int = 0
    llm_base_url: str | None = None
    llm_model: str | None = None
    multi_step: bool = True
    overwrite_output: bool = False
    concurrency: int = 1
