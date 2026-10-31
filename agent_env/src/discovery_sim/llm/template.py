from __future__ import annotations

from typing import Any

from discovery_sim.llm.base import LlmPlanner, PlannedToolCall, PlannerResult
from discovery_sim.sim.context import AgentContext


class TemplatePlanner(LlmPlanner):
    """Deterministic planner (no API). Uses step_hint from the scenario."""

    model_id = "template"

    async def plan(self, context: AgentContext, step_hint: dict[str, Any] | None = None) -> PlannerResult:
        if step_hint is None:
            return PlannerResult(call=None, error="no step_hint")
        call = PlannedToolCall(
            agent=step_hint["agent"],
            tool_name=step_hint["tool_name"],
            arguments=dict(step_hint["arguments"]),
        )
        return PlannerResult(call=call, model_id=self.model_id)
