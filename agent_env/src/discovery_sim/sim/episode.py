from __future__ import annotations

import json
import random
import uuid
from pathlib import Path
from typing import Any
from tqdm import tqdm

from discovery_sim.llm.base import LlmPlanner
from discovery_sim.llm.chat import ChatClient, LmStudioChat
from discovery_sim.llm.lmstudio import LmStudioPlanner
from discovery_sim.llm.template import TemplatePlanner
from discovery_sim.ledger.store import SyntheticLedger
from discovery_sim.sim.agent_tasks import agent_tasks
from discovery_sim.sim.context import AgentContext
from discovery_sim.sim.harness import AgentPool, execute_tool_call
from discovery_sim.sim.labels import label_call, laya_state
from discovery_sim.sim.operators import operator_catalog, pick_operator
from discovery_sim.sim.fault_schedule import pick_agent_fault_call_index
from discovery_sim.sim.orchestrator import AgentOrchestrator, agent_episode_outcome
from discovery_sim.sim.run_config import RunConfig
from discovery_sim.sim.scenarios import EpisodeScenario, all_scenarios, tool_call_from_planned
from discovery_sim.sim.types import AttackSpec, EpisodeOutcome, LabeledExample, OperatorId, ToolCall


LAYA_QUESTIONS = {
    "output_correct": {
        "type": "noul",
        "instructions": (
            "Given the task, the model tool call, and the arguments the tool received, "
            "does the output look correct? Answer true when it matches; false when it does not."
        ),
        "labels": {"false": "output does not match", "true": "output matches"},
    },
    "cause": {
        "type": "choice",
        "instructions": "If the output is wrong, is the fault from the agent plan or an adversary?",
        "criteria": {
            "none": "output is correct",
            "agent": "model call was wrong and tool obeyed it",
            "adversary": "model call was right but args or output were tampered",
        },
    },
}


class EpisodeRunner:
    def __init__(
        self,
        seed: int = 0,
        config: RunConfig | None = None,
        chat_client: ChatClient | None = None,
    ) -> None:
        self.rng = random.Random(seed)
        self.seed = seed
        self.config = config or RunConfig(seed=seed)
        self._chat_client = chat_client

    def _chat_for_episode(self) -> ChatClient:
        if self._chat_client is not None:
            return self._chat_client
        return LmStudioChat(base_url=self.config.llm_base_url, model=self.config.llm_model)

    def _planner_for_episode(self, episode_index: int) -> LlmPlanner:
        if episode_index < self.config.llm_samples:
            return LmStudioPlanner(base_url=self.config.llm_base_url, model=self.config.llm_model)
        return TemplatePlanner()

    async def run_one(
        self,
        operator_override: OperatorId | None = None,
        scenario_index: int | None = None,
        episode_index: int = 0,
        fault_call_index_override: int | None = None,
    ) -> EpisodeOutcome | None:
        if self.config.mode == "agent":
            return await self._run_one_agent(
                operator_override, scenario_index, episode_index, fault_call_index_override
            )
        return await self._run_one_scripted(operator_override, scenario_index, episode_index)

    async def _run_one_agent(
        self,
        operator_override: OperatorId | None = None,
        scenario_index: int | None = None,
        episode_index: int = 0,
        fault_call_index_override: int | None = None,
    ) -> EpisodeOutcome | None:
        tasks = agent_tasks()
        idx = scenario_index if scenario_index is not None else self.rng.randrange(len(tasks))
        task = tasks[idx]
        catalog = operator_catalog()
        attack = catalog[operator_override] if operator_override else pick_operator(self.rng)
        if attack.operator == OperatorId.TAINT_PRIOR_OUTPUT:
            attack = catalog[OperatorId.BENIGN]
        if fault_call_index_override is not None:
            fault_call_index = fault_call_index_override
        else:
            fault_call_index = pick_agent_fault_call_index(
                task, attack, self.rng, self.config.max_agent_steps
            )

        pool = AgentPool(SyntheticLedger())
        orchestrator = AgentOrchestrator(self._chat_for_episode(), max_steps=self.config.max_agent_steps)
        run = await orchestrator.run_episode(task, pool, attack, fault_call_index, self.seed)
        if not run.records and not run.task_completed:
            return None
        return agent_episode_outcome(task, attack, fault_call_index, self.seed, run)

    async def _run_one_scripted(
        self,
        operator_override: OperatorId | None = None,
        scenario_index: int | None = None,
        episode_index: int = 0,
    ) -> EpisodeOutcome | None:
        scenarios = all_scenarios(include_multi=self.config.multi_step)
        idx = scenario_index if scenario_index is not None else self.rng.randrange(len(scenarios))
        scenario = scenarios[idx]
        scenario = _clone_scenario(scenario)

        catalog = operator_catalog()
        if operator_override is not None:
            attack = catalog[operator_override]
        else:
            attack = pick_operator(self.rng)

        if attack.operator == OperatorId.TAINT_PRIOR_OUTPUT and len(scenario.steps) < 2:
            attack = catalog[OperatorId.BENIGN]

        fault_call_index = self.rng.randrange(len(scenario.steps)) if attack.operator != OperatorId.BENIGN else 0

        pool = AgentPool(SyntheticLedger())
        context = scenario.context or AgentContext.from_mandate(scenario.user_message, scenario.mandate)
        planner = self._planner_for_episode(episode_index)
        use_live_llm = episode_index < self.config.llm_samples

        records = []
        examples = []
        failure_call_index: int | None = None

        for step_index, step in enumerate(scenario.steps):
            step_attack = attack if step_index == fault_call_index else catalog[OperatorId.BENIGN]
            task = step.to_base_task(scenario, step_index)

            llm_call = await self._plan_call(
                planner,
                context,
                step,
                task,
                step_attack,
                use_live_llm,
            )
            if llm_call is None:
                return None

            record = await execute_tool_call(
                pool,
                task,
                llm_call,
                step_attack,
                call_index=step_index,
                context=context,
            )
            labels = label_call(task, record, pool.ledger)
            context.record_call(step_index, record, labels.output_correct)

            examples.append(
                LabeledExample(
                    example_id=f"ex_{uuid.uuid4().hex[:8]}",
                    input_state={
                        "state_text": laya_state(task, record, context, pool.ledger),
                        "base_task": scenario.user_message,
                        "trusted": dict(context.trusted),
                        "prior_calls": [
                            {
                                "call_index": p.call_index,
                                "tool": p.llm_call.tool_name,
                                "model_call": p.llm_call.arguments,
                                "output": p.output,
                            }
                            for p in context.prior_calls
                            if p.call_index < step_index
                        ],
                        "model_call": record.llm_call.arguments,
                        "received_args": record.mcp_arguments,
                        "output": record.output,
                        "ledger_snapshot": context.ledger_snapshot(pool.ledger),
                    },
                    labels=labels,
                    meta={
                        "operator": step_attack.operator.value,
                        "hop": step_attack.hop,
                        "seed": self.seed,
                        "call_index": step_index,
                        "mode": "scripted",
                        "planner": getattr(planner, "model_id", "unknown"),
                        "live_llm": use_live_llm and step_attack.operator == OperatorId.BENIGN,
                        "hidden_adversary": record.hidden_adversary,
                        "fault": step_attack.fault,
                        "taint_map": dict(context.taint_map),
                    },
                )
            )
            records.append(record)

            if record.error or not record.output:
                failure_call_index = step_index
                break

        last_labels = examples[-1].labels if examples else None
        task_completed = bool(records) and records[-1].error is None and bool(records[-1].output)
        task_correct = task_completed and last_labels is not None and last_labels.output_correct
        if len(examples) > 1:
            task_correct = task_completed and all(ex.labels.output_correct for ex in examples)
        silent_wrong = task_completed and not task_correct
        hard_fail = not task_completed
        absorbed = attack.operator != OperatorId.BENIGN and task_correct and task_completed

        fault_record = records[min(fault_call_index, len(records) - 1)]
        attack_applied = attack.operator != OperatorId.BENIGN and any(
            r.attack.operator != OperatorId.BENIGN for r in records
        )
        return EpisodeOutcome(
            episode_id=f"ep_{uuid.uuid4().hex[:8]}",
            seed=self.seed,
            operator=attack.operator,
            task_completed=task_completed,
            task_correct=task_correct,
            silent_wrong=silent_wrong,
            hard_fail=hard_fail,
            absorbed=absorbed,
            fault_agent=fault_record.llm_call.agent if records else "bank",
            fault_tool=fault_record.llm_call.tool_name if records else "",
            fault_hop=attack.hop,
            fault_call_index=fault_call_index,
            failure_call_index=failure_call_index,
            attack_applied=attack_applied,
            records=records,
            examples=examples,
        )

    async def _plan_call(
        self,
        planner: LlmPlanner,
        context: AgentContext,
        step,
        task,
        attack: AttackSpec,
        use_live_llm: bool,
    ) -> ToolCall | None:
        from discovery_sim.sim.catalog import tool_call_for_task

        if use_live_llm and attack.operator == OperatorId.BENIGN:
            result = await planner.plan(context, step_hint=None)
            if result.error or result.call is None:
                return None
            planned = result.call
            expected = step.resolve_arguments(context)
            if (
                planned.tool_name != step.tool_name
                or planned.agent != step.target_agent
                or planned.arguments != expected
            ):
                return None
            return tool_call_from_planned(planned)

        hint = step.planner_hint(context)
        result = await TemplatePlanner().plan(context, step_hint=hint)
        if result.call is None:
            return None
        llm_call = tool_call_from_planned(result.call)
        if attack.hop == "llm" and attack.operator in (
            OperatorId.LLM_WRONG_ACCOUNT,
            OperatorId.LLM_WRONG_AMOUNT,
        ):
            from discovery_sim.sim.operators import apply_llm_attack

            llm_call = apply_llm_attack(task, llm_call, attack.operator)
        return llm_call


def _clone_scenario(scenario: EpisodeScenario) -> EpisodeScenario:
    return EpisodeScenario(
        scenario_id=scenario.scenario_id,
        template=scenario.template,
        user_message=scenario.user_message,
        mandate=dict(scenario.mandate),
        steps=scenario.steps,
        context=AgentContext.from_mandate(scenario.user_message, scenario.mandate),
    )


async def run_episodes(config: RunConfig, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = EpisodeRunner(seed=config.seed, config=config)
    episodes: list[EpisodeOutcome] = []
    dropped_llm = 0
    for i in tqdm(range(config.episodes)):
        ep = await runner.run_one(episode_index=i)
        if ep is None:
            dropped_llm += 1
            continue
        episodes.append(ep)

    episodes_path = out_dir / "episodes.jsonl"
    examples_path = out_dir / "examples.jsonl"
    laya_path = out_dir / "laya_train.jsonl"
    questions_path = out_dir / "questions.json"

    with episodes_path.open("w") as ef, examples_path.open("w") as xf, laya_path.open("w") as lf:
        for ep in episodes:
            ef.write(json.dumps(_episode_to_json(ep)) + "\n")
            for ex in ep.examples:
                xf.write(json.dumps(_example_to_json(ex)) + "\n")
                lf.write(
                    json.dumps(
                        {
                            "state": ex.input_state["state_text"],
                            "questions": LAYA_QUESTIONS,
                            "expected": {
                                "output_correct": ex.labels.output_correct,
                                "cause": ex.labels.cause,
                            },
                        }
                    )
                    + "\n"
                )

    questions_path.write_text(json.dumps(LAYA_QUESTIONS, indent=2))
    baseline = _baseline_from_episodes(episodes, config.seed, config.episodes)
    baseline["mode"] = config.mode
    baseline["dropped_llm_mismatch"] = dropped_llm
    baseline["llm_samples_requested"] = config.llm_samples
    baseline["max_agent_steps"] = config.max_agent_steps
    (out_dir / "baseline.json").write_text(json.dumps(baseline, indent=2))
    return {
        "episodes": len(episodes),
        "requested": config.episodes,
        "dropped_llm": dropped_llm,
        "out_dir": str(out_dir),
        "baseline": baseline,
    }


def _episode_to_json(ep: EpisodeOutcome) -> dict[str, Any]:
    return {
        "episode_id": ep.episode_id,
        "seed": ep.seed,
        "operator": ep.operator.value,
        "task_completed": ep.task_completed,
        "task_correct": ep.task_correct,
        "silent_wrong": ep.silent_wrong,
        "hard_fail": ep.hard_fail,
        "absorbed": ep.absorbed,
        "fault_agent": ep.fault_agent,
        "fault_tool": ep.fault_tool,
        "fault_hop": ep.fault_hop,
        "fault_call_index": ep.fault_call_index,
        "failure_call_index": ep.failure_call_index,
        "attack_applied": ep.attack_applied,
        "num_calls": len(ep.records),
        "mode": ep.examples[0].meta.get("mode") if ep.examples else None,
    }


def _example_to_json(ex: LabeledExample) -> dict[str, Any]:
    return {
        "example_id": ex.example_id,
        "input": ex.input_state,
        "label": {
            "output_correct": ex.labels.output_correct,
            "cause": ex.labels.cause,
            "verdict": ex.labels.verdict,
            "fault": ex.labels.fault,
            "severity": ex.labels.severity,
        },
        "meta": ex.meta,
    }


def _baseline_from_episodes(episodes: list[EpisodeOutcome], seed: int, count: int) -> dict[str, Any]:
    attacked = [e for e in episodes if e.operator != OperatorId.BENIGN]
    applied = [e for e in attacked if e.attack_applied]
    n = len(attacked) or 1
    n_applied = len(applied) or 1
    return {
        "seed": seed,
        "episode_count": count,
        "episodes_written": len(episodes),
        "attack_applied_rate": sum(1 for e in attacked if e.attack_applied) / n,
        "task_correct_rate": sum(1 for e in attacked if e.task_correct) / n,
        "silent_wrong_rate": sum(1 for e in attacked if e.silent_wrong) / n,
        "hard_fail_rate": sum(1 for e in attacked if e.hard_fail) / n,
        "task_correct_rate_applied_only": sum(1 for e in applied if e.task_correct) / n_applied,
        "silent_wrong_rate_applied_only": sum(1 for e in applied if e.silent_wrong) / n_applied,
        "by_operator": _group_rates(attacked, lambda e: e.operator.value),
        "by_hop": _group_rates(attacked, lambda e: e.fault_hop or "none"),
    }


def _group_rates(episodes: list[EpisodeOutcome], key_fn) -> dict[str, Any]:
    groups: dict[str, list[EpisodeOutcome]] = {}
    for ep in episodes:
        groups.setdefault(key_fn(ep), []).append(ep)
    out: dict[str, Any] = {}
    for name, items in groups.items():
        n = len(items)
        out[name] = {
            "count": n,
            "silent_wrong_rate": sum(1 for e in items if e.silent_wrong) / n,
            "hard_fail_rate": sum(1 for e in items if e.hard_fail) / n,
        }
    return out
