from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from discovery_sim.sim.context import AgentContext
from discovery_sim.sim.types import BaseTask, ToolCall


@dataclass
class StepSpec:
    target_agent: Literal["bank", "payment"]
    tool_name: str
    arguments: dict[str, Any] | Callable[[AgentContext], dict[str, Any]]

    def resolve_arguments(self, context: AgentContext) -> dict[str, Any]:
        if callable(self.arguments):
            return dict(self.arguments(context))
        return dict(self.arguments)

    def to_base_task(self, scenario: EpisodeScenario, step_index: int) -> BaseTask:
        args = self.resolve_arguments(scenario.context)
        return BaseTask(
            task_id=f"{scenario.scenario_id}_step{step_index}",
            template=scenario.template,
            user_message=scenario.user_message,
            mandate=dict(scenario.mandate),
            target_agent=self.target_agent,
            tool_name=self.tool_name,
            correct_arguments=args,
        )

    def planner_hint(self, context: AgentContext) -> dict[str, Any]:
        return {
            "agent": self.target_agent,
            "tool_name": self.tool_name,
            "arguments": self.resolve_arguments(context),
        }


@dataclass
class EpisodeScenario:
    scenario_id: str
    template: str
    user_message: str
    mandate: dict[str, Any]
    steps: list[StepSpec]
    context: AgentContext | None = None

    def __post_init__(self) -> None:
        if self.context is None:
            self.context = AgentContext.from_mandate(self.user_message, self.mandate)


def single_call_scenarios() -> list[EpisodeScenario]:
    from discovery_sim.sim.catalog import task_templates

    out: list[EpisodeScenario] = []
    for t in task_templates():
        out.append(
            EpisodeScenario(
                scenario_id=t.task_id,
                template=t.template,
                user_message=t.user_message,
                mandate=dict(t.mandate),
                steps=[
                    StepSpec(
                        target_agent=t.target_agent,
                        tool_name=t.tool_name,
                        arguments=dict(t.correct_arguments),
                    )
                ],
            )
        )
    return out


def multi_step_scenarios() -> list[EpisodeScenario]:
    return [
        EpisodeScenario(
            scenario_id="balance_then_transfer",
            template="balance_then_transfer",
            user_message="Check balance on acct_synth_0031, then pay payee_synth_0007 $50 from that account.",
            mandate={
                "account_id": "acct_synth_0031",
                "from_account_id": "acct_synth_0031",
                "payee_id": "payee_synth_0007",
                "amount_cents": 5000,
                "cap_cents": 100_000,
            },
            steps=[
                StepSpec(
                    target_agent="bank",
                    tool_name="get_balance",
                    arguments={"account_id": "acct_synth_0031"},
                ),
                StepSpec(
                    target_agent="payment",
                    tool_name="initiate_transfer",
                    arguments=lambda ctx: {
                        "from_account_id": "acct_synth_0031",
                        "payee_id": "payee_synth_0007",
                        "amount_cents": 5000,
                    },
                ),
            ],
        ),
        EpisodeScenario(
            scenario_id="balance_then_transfer_alt",
            template="balance_then_transfer",
            user_message="Check acct_synth_0042, then pay payee_synth_0007 $25 from that account.",
            mandate={
                "account_id": "acct_synth_0042",
                "from_account_id": "acct_synth_0042",
                "payee_id": "payee_synth_0007",
                "amount_cents": 2500,
                "cap_cents": 100_000,
            },
            steps=[
                StepSpec(
                    target_agent="bank",
                    tool_name="get_balance",
                    arguments={"account_id": "acct_synth_0042"},
                ),
                StepSpec(
                    target_agent="payment",
                    tool_name="initiate_transfer",
                    arguments={
                        "from_account_id": "acct_synth_0042",
                        "payee_id": "payee_synth_0007",
                        "amount_cents": 2500,
                    },
                ),
            ],
        ),
    ]


def all_scenarios(include_multi: bool = True) -> list[EpisodeScenario]:
    scenarios = single_call_scenarios()
    if include_multi:
        scenarios.extend(multi_step_scenarios())
    return scenarios


def scenario_index(scenario_id: str, *, include_multi: bool = True) -> int:
    for i, scenario in enumerate(all_scenarios(include_multi=include_multi)):
        if scenario.scenario_id == scenario_id:
            return i
    raise KeyError(scenario_id)


def tool_call_from_planned(planned) -> ToolCall:
    return ToolCall(
        tool_name=planned.tool_name,
        arguments=dict(planned.arguments),
        agent=planned.agent,
    )
