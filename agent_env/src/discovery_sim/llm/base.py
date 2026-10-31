from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from discovery_sim.sim.context import AgentContext


@dataclass
class PlannedToolCall:
    agent: Literal["bank", "payment"]
    tool_name: str
    arguments: dict[str, Any]


@dataclass
class PlannerResult:
    call: PlannedToolCall | None
    raw_text: str | None = None
    model_id: str | None = None
    error: str | None = None


class LlmPlanner(ABC):
    @abstractmethod
    async def plan(self, context: "AgentContext", step_hint: dict[str, Any] | None = None) -> PlannerResult:
        raise NotImplementedError
