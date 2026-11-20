from __future__ import annotations

import asyncio
import json
import random
import traceback
import uuid
from pathlib import Path
from typing import Any

from mcp_hardening import GenAITelemetry, HardeningRuntime, semconv
from mcp_hardening.config import TelemetryConfig
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from discovery_sim.llm.base import LlmPlanner
from discovery_sim.llm.chat import ChatClient, load_dotenv, make_chat_client
from discovery_sim.llm.lmstudio import LmStudioPlanner
from discovery_sim.llm.template import TemplatePlanner
from discovery_sim.ledger.store import SyntheticLedger
from discovery_sim.sim.agent_tasks import agent_tasks
from discovery_sim.sim.context import AgentContext
from discovery_sim.sim.harness import AgentPool, execute_tool_call
from discovery_sim.sim.labels import example_input_state, label_call
from discovery_sim.sim.operators import operator_catalog, pick_operator
from discovery_sim.sim.fault_schedule import pick_agent_fault_call_index
from discovery_sim.sim.orchestrator import (
    AgentOrchestrator,
    ToolCallError,
    agent_episode_outcome,
    tool_error_type,
)
from discovery_sim.sim.run_config import RunConfig
from discovery_sim.sim.run_output import resolve_run_dir, write_run_manifest
from discovery_sim.sim.scenarios import EpisodeScenario, all_scenarios, tool_call_from_planned
from discovery_sim.sim.task_defs import task_sets
from discovery_sim.sim.task_sample import choose_agent_task
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


# Files every run writes.
EPISODES_FILE = "episodes.jsonl"
EXAMPLES_FILE = "examples.jsonl"
LAYA_TRAIN_FILE = "laya_train.jsonl"
QUESTIONS_FILE = "questions.json"
BASELINE_FILE = "baseline.json"
# Simulator crashes (our bugs) vs. tool calls the model malformed (model failures); written only if any.
ERRORS_FILE = "errors.jsonl"
MALFORMED_CALLS_FILE = "malformed_calls.jsonl"


class EpisodeError(Exception):
    def __init__(self, details: dict[str, Any]) -> None:
        super().__init__(f"episode failed on task {details.get('task_id')}")
        self.details = details


def _error_record(episode_index: int, exc: Exception) -> dict[str, Any]:
    details = exc.details if isinstance(exc, EpisodeError) else {}
    # Report the innermost exception (e.g. the KeyError), not our wrappers.
    root: BaseException = exc
    while isinstance(root, (EpisodeError, ToolCallError)) and root.__cause__ is not None:
        root = root.__cause__
    return {
        "episode_index": episode_index,
        **details,
        "error_type": type(root).__name__,
        "error": str(root),
        "traceback": "".join(traceback.format_exception(type(root), root, root.__traceback__)),
    }


class EpisodeRunner:
    def __init__(
        self,
        seed: int = 0,
        config: RunConfig | None = None,
        chat_client: ChatClient | None = None,
        runtime: HardeningRuntime | None = None,
        telemetry: GenAITelemetry | None = None,
    ) -> None:
        self.rng = random.Random(seed)
        self.seed = seed
        self.config = config or RunConfig(seed=seed)
        self._chat_client = chat_client
        self.store = None  # a state_db.StateStore when the run records state (``RunConfig.state_db``)
        self.panel = None  # a recovery_classifiers.Panel when the run lists classifiers (``RunConfig.recovery_classifiers``)
        # MCP hardening middleware shared by every episode (one nonce cache, one key).
        self.runtime = runtime or HardeningRuntime.build(self.config.hardening)
        self.telemetry = telemetry or GenAITelemetry(TelemetryConfig(enabled=False))

    @staticmethod
    def _parallel_episode_rng(base_seed: int, episode_index: int) -> random.Random:
        return random.Random(base_seed + episode_index * 1_000_003)

    def _roll_rng(self, episode_rng: random.Random | None) -> random.Random:
        return episode_rng if episode_rng is not None else self.rng

    def _chat_for_episode(self) -> ChatClient:
        if self._chat_client is not None:
            return self._chat_client
        return make_chat_client(self.config.llm_backend, base_url=self.config.llm_base_url, model=self.config.llm_model,
                                temperature=self.config.temperature)

    def _planner_for_episode(self, episode_index: int) -> LlmPlanner:
        if episode_index < self.config.llm_samples:
            return LmStudioPlanner(base_url=self.config.llm_base_url, model=self.config.llm_model)
        return TemplatePlanner()

    def _allowed_operators(self) -> list[OperatorId] | None:
        if not self.config.operators:
            return None
        return [OperatorId(name) for name in self.config.operators]

    def _operator_weights(self) -> dict[OperatorId, float] | None:
        if not self.config.operator_weights:
            return None
        return {OperatorId(name): float(weight) for name, weight in self.config.operator_weights.items()}

    def _pick_attack(self, rng: random.Random, operator_override: OperatorId | None) -> AttackSpec:
        catalog = operator_catalog()
        if operator_override is not None:
            return catalog[operator_override]
        return pick_operator(
            rng,
            allowed=self._allowed_operators(),
            weights=self._operator_weights(),
        )

    async def run_one(
        self,
        operator_override: OperatorId | None = None,
        scenario_index: int | None = None,
        episode_index: int = 0,
        fault_call_index_override: int | None = None,
        episode_rng: random.Random | None = None,
    ) -> EpisodeOutcome | None:
        episode_id = f"ep_{uuid.uuid4().hex[:8]}"
        if self.config.mode == "agent":
            return await self._run_one_agent(
                operator_override,
                scenario_index,
                episode_index,
                fault_call_index_override,
                episode_rng,
                episode_id=episode_id,
            )
        with self.telemetry.agent_span(
            "scripted-agent", episode_id, {semconv.SIM_EPISODE_INDEX: episode_index}
        ) as span:
            ep = await self._run_one_scripted(
                operator_override, scenario_index, episode_index, episode_rng, episode_id=episode_id
            )
            if ep is not None:
                self.telemetry.end_agent(
                    span,
                    completed=ep.task_completed,
                    error=next((r.error for r in reversed(ep.records) if r.error), None),
                    attributes={
                        semconv.SIM_OPERATOR: ep.operator.value,
                        semconv.SIM_TASK_COMPLETED: ep.task_completed,
                        semconv.SIM_TASK_CORRECT: ep.task_correct,
                    },
                )
            return ep

    async def _run_one_agent(
        self,
        operator_override: OperatorId | None = None,
        scenario_index: int | None = None,
        episode_index: int = 0,
        fault_call_index_override: int | None = None,
        episode_rng: random.Random | None = None,
        episode_id: str | None = None,
    ) -> EpisodeOutcome | None:
        rng = self._roll_rng(episode_rng)
        allowed = self._allowed_operators()
        task = choose_agent_task(
            rng,
            task_set=self.config.task_set,
            task_sample=self.config.task_sample,
            scenario_index=scenario_index,
            operators=allowed,
        )
        catalog = operator_catalog()
        attack = self._pick_attack(rng, operator_override)
        if attack.operator == OperatorId.TAINT_PRIOR_OUTPUT:
            attack = catalog[OperatorId.BENIGN]
        if fault_call_index_override is not None:
            fault_call_index = fault_call_index_override
        else:
            fault_call_index = pick_agent_fault_call_index(
                task, attack, rng, self.config.max_agent_steps
            )

        pool = AgentPool(SyntheticLedger(), runtime=self.runtime)
        orchestrator = AgentOrchestrator(
            self._chat_for_episode(),
            max_steps=self.config.max_agent_steps,
            store=self.store,
            panel=self.panel,
            telemetry=self.telemetry,
        )
        try:
            run = await orchestrator.run_episode(
                task, pool, attack, fault_call_index, self.seed, rng=rng, episode_id=episode_id
            )
        except Exception as exc:
            details: dict[str, Any] = {
                "task_id": task.task_id,
                "template": task.template,
                "user_message": task.user_message,
                "mandate": task.mandate,
                "operator": attack.operator.value,
                "fault_call_index": fault_call_index,
            }
            if isinstance(exc, ToolCallError):
                details["tool_call"] = {
                    "call_index": exc.call_index,
                    "tool_name": exc.tool_name,
                    "arguments": exc.arguments,
                }
                details["conversation"] = exc.conversation
            raise EpisodeError(details) from exc
        if not run.records and not run.task_completed and not run.malformed_calls:
            return None
        return agent_episode_outcome(task, attack, fault_call_index, self.seed, run)

    async def _run_one_scripted(
        self,
        operator_override: OperatorId | None = None,
        scenario_index: int | None = None,
        episode_index: int = 0,
        episode_rng: random.Random | None = None,
        episode_id: str | None = None,
    ) -> EpisodeOutcome | None:
        rng = self._roll_rng(episode_rng)
        scenarios = all_scenarios(include_multi=self.config.multi_step)
        idx = scenario_index if scenario_index is not None else rng.randrange(len(scenarios))
        scenario = scenarios[idx]
        scenario = _clone_scenario(scenario)

        catalog = operator_catalog()
        attack = self._pick_attack(rng, operator_override)

        if attack.operator == OperatorId.TAINT_PRIOR_OUTPUT and len(scenario.steps) < 2:
            attack = catalog[OperatorId.BENIGN]

        fault_call_index = rng.randrange(len(scenario.steps)) if attack.operator != OperatorId.BENIGN else 0

        pool = AgentPool(SyntheticLedger(), runtime=self.runtime)
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

            with self.telemetry.tool_span(
                llm_call.tool_name,
                f"{task.task_id}_call",
                arguments=llm_call.arguments,
                attributes={
                    semconv.SIM_CALL_INDEX: step_index,
                    semconv.SIM_MALFORMED: False,
                    semconv.SIM_OPERATOR: step_attack.operator.value,
                    semconv.SIM_HOP: step_attack.hop,
                },
            ) as tool_span:
                record = await execute_tool_call(
                    pool,
                    task,
                    llm_call,
                    step_attack,
                    call_index=step_index,
                    context=context,
                    rng=rng,
                )
                labels = label_call(task, record, pool.ledger)
                self.telemetry.end_tool(
                    tool_span,
                    result=record.output,
                    error_type=tool_error_type(record),
                    error=record.error,
                    attributes={
                        semconv.SIM_OUTPUT_CORRECT: labels.output_correct,
                        semconv.SIM_CAUSE: labels.cause,
                    },
                )
            context.record_call(step_index, record, labels.output_correct)

            examples.append(
                LabeledExample(
                    example_id=f"ex_{uuid.uuid4().hex[:8]}",
                    input_state=example_input_state(task, record, context, pool.ledger),
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
                        "hardening": dict(record.hardening),
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
            episode_id=episode_id or f"ep_{uuid.uuid4().hex[:8]}",
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
        # LLM-hop mutations are applied once in execute_tool_call (same path as agent mode).
        return tool_call_from_planned(result.call)


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
    """Run ``config.episodes`` episodes and write every output file into ``out_dir``."""
    out_dir = resolve_run_dir(out_dir, config.seed, overwrite=config.overwrite_output)
    out_dir.mkdir(parents=True, exist_ok=True)
    load_dotenv()  # encrypted MCP hardening key lives in agent_env/.env
    runtime = HardeningRuntime.build(config.hardening)
    telemetry = GenAITelemetry(config.hardening.telemetry, out_dir)
    try:
        return await _run_episodes(config, out_dir, runtime, telemetry)
    finally:
        # Flush the last telemetry part file even if the run crashes.
        telemetry.shutdown()


async def _run_episodes(
    config: RunConfig,
    out_dir: Path,
    runtime: HardeningRuntime,
    telemetry: GenAITelemetry,
) -> dict[str, Any]:
    workers = max(1, config.concurrency)

    store = _open_state_db(out_dir) if config.state_db else None
    try:
        episodes, errors, dropped_llm = await _run_all_episodes(
            config, workers, store, runtime=runtime, telemetry=telemetry
        )
    finally:
        if store is not None:
            store.close()

    _write_failure_files(out_dir, episodes, errors)
    _write_episode_files(out_dir, episodes)

    baseline = _baseline_from_episodes(episodes, config.seed, config.episodes)
    baseline["mode"] = config.mode
    baseline["dropped_llm_mismatch"] = dropped_llm
    baseline["episode_errors"] = len(errors)
    baseline["llm_samples_requested"] = config.llm_samples
    baseline["max_agent_steps"] = config.max_agent_steps
    baseline["concurrency"] = workers
    baseline["hardening"] = config.hardening.to_dict()
    baseline["hardening_blocked_calls"] = _hardening_blocked(episodes)
    planners = sorted(
        {str(ex.meta["planner"]) for ep in episodes for ex in ep.examples if ex.meta.get("planner")}
    )
    baseline["models"] = planners
    baseline["model"] = planners[0] if len(planners) == 1 else None
    (out_dir / BASELINE_FILE).write_text(json.dumps(baseline, indent=2))
    write_run_manifest(out_dir, _run_manifest(config, workers, planners, out_dir, runtime, telemetry))

    return {
        "episodes": len(episodes),
        "requested": config.episodes,
        "dropped_llm": dropped_llm,
        "errors": len(errors),
        "model_failures": baseline["model_failure_episodes"],
        "concurrency": workers,
        "out_dir": str(out_dir),
        "baseline": baseline,
    }


def _open_state_db(out_dir: Path):
    """A fresh ``state.db`` for this run; imported here so the package stays optional."""
    try:
        from state_db import StateStore
    except ImportError as exc:
        raise SystemExit("--state-db needs the state_db package: agent_env/.venv/bin/pip install -e ../state_db") from exc
    for name in ("state.db",):
        (out_dir / name).unlink(missing_ok=True)  # a rerun into the same folder starts a new record
    return StateStore(out_dir / "state.db")


def _build_classifiers(specs):
    """Build and fit the run's classifiers once; imported here so the package stays optional."""
    try:
        from recovery_classifiers import Panel
    except ImportError as exc:
        raise SystemExit("classifiers need the recovery_classifiers package: "
                         "agent_env/.venv/bin/pip install -e ../mlp_baseline -e ../recovery_classifiers") from exc
    return Panel(specs)


async def _run_all_episodes(
    config: RunConfig,
    workers: int,
    store=None,
    runtime: HardeningRuntime | None = None,
    telemetry: GenAITelemetry | None = None,
) -> tuple[list[EpisodeOutcome], list[dict[str, Any]], int]:
    """Run every episode (in parallel when ``workers > 1``), in episode order.

    Returns the finished episodes, one error record per crashed episode, and how many
    episodes were dropped for an LM plan mismatch.
    """
    runner = EpisodeRunner(seed=config.seed, config=config, runtime=runtime, telemetry=telemetry)
    runner.store = store
    runner.panel = _build_classifiers(config.recovery_classifiers) if config.recovery_classifiers else None
    errors: list[dict[str, Any]] = []

    async def _run_episode_index(i: int) -> tuple[int, EpisodeOutcome | None]:
        try:
            if workers > 1:
                ep_rng = EpisodeRunner._parallel_episode_rng(config.seed, i)
                ep = await runner.run_one(episode_index=i, episode_rng=ep_rng)
            else:
                ep = await runner.run_one(episode_index=i)
        except Exception as exc:
            # One crashing episode must not discard the rest of the run.
            record = _error_record(i, exc)
            errors.append(record)
            tqdm.write(
                f"episode {i} failed ({record.get('task_id', '?')}): "
                f"{record['error_type']}: {record['error']}"
            )
            return i, None
        return i, ep

    if workers > 1:
        sem = asyncio.Semaphore(workers)

        async def _limited(i: int) -> tuple[int, EpisodeOutcome | None]:
            async with sem:
                return await _run_episode_index(i)

        indexed = await tqdm_asyncio.gather(
            *[_limited(i) for i in range(config.episodes)],
            desc="episodes",
        )
        indexed.sort(key=lambda pair: pair[0])
    else:
        indexed = [await _run_episode_index(i) for i in tqdm(range(config.episodes), desc="episodes")]

    episodes = [ep for _i, ep in indexed if ep is not None]
    # Crashed episodes also come back as None; don't count them as LM-plan mismatches.
    dropped_llm = len(indexed) - len(episodes) - len(errors)
    errors.sort(key=lambda rec: rec["episode_index"])
    return episodes, errors, dropped_llm


def _write_failure_files(out_dir: Path, episodes: list[EpisodeOutcome], errors: list[dict[str, Any]]) -> None:
    """Simulator crashes and model-malformed calls, each written only when there are any."""
    if errors:
        _write_jsonl(out_dir / ERRORS_FILE, errors)

    malformed = [
        {"episode_id": ep.episode_id, "operator": ep.operator.value, "recovered": not ep.model_failure, **call}
        for ep in episodes
        for call in ep.malformed_calls
    ]
    if malformed:
        _write_jsonl(out_dir / MALFORMED_CALLS_FILE, malformed)


def _write_episode_files(out_dir: Path, episodes: list[EpisodeOutcome]) -> None:
    """One row per episode, one row per tool call (full and Laya format), and the question schema."""
    with (
        (out_dir / EPISODES_FILE).open("w") as ef,
        (out_dir / EXAMPLES_FILE).open("w") as xf,
        (out_dir / LAYA_TRAIN_FILE).open("w") as lf,
    ):
        for ep in episodes:
            ef.write(json.dumps(_episode_to_json(ep)) + "\n")
            for ex in ep.examples:
                xf.write(json.dumps(_example_to_json(ex)) + "\n")
                lf.write(json.dumps(_laya_row(ex)) + "\n")

    (out_dir / QUESTIONS_FILE).write_text(json.dumps(LAYA_QUESTIONS, indent=2))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _run_manifest(
    config: RunConfig,
    workers: int,
    planners: list[str],
    out_dir: Path,
    runtime: HardeningRuntime,
    telemetry: GenAITelemetry,
) -> dict[str, Any]:
    return {
        "seed": config.seed,
        "episodes": config.episodes,
        "mode": config.mode,
        "model": planners[0] if len(planners) == 1 else None,
        "models": planners,
        "max_agent_steps": config.max_agent_steps,
        "multi_step": config.multi_step,
        "llm_samples": config.llm_samples,
        "concurrency": workers,
        "temperature": config.temperature,
        "task_set": config.task_set,
        "task_sample": config.task_sample,
        "task_count": len(agent_tasks(config.task_set)) if config.task_set in task_sets() else None,
        "operators": list(config.operators) if config.operators else None,
        "operator_weights": dict(config.operator_weights) if config.operator_weights else None,
        "hardening": config.hardening.to_dict(),
        "hardening_key": (
            {"kid": runtime.keyring.kid, "ephemeral": runtime.keyring.ephemeral}
            if runtime.keyring is not None
            else None
        ),
        "telemetry_dir": str(telemetry.out_dir) if telemetry.out_dir else None,
        "out_dir": str(out_dir),
        # Only present when on, so manifests of ordinary runs are unchanged.
        **({"state_db": "state.db"} if config.state_db else {}),
        **({"recovery_classifiers": list(config.recovery_classifiers)} if config.recovery_classifiers else {}),
    }


def _laya_row(ex: LabeledExample) -> dict[str, Any]:
    expected = {"output_correct": ex.labels.output_correct, "cause": ex.labels.cause}
    return {"state": ex.input_state["state_text"], "questions": LAYA_QUESTIONS, "expected": expected}


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
        "malformed_calls": len(ep.malformed_calls),
        "model_failure": ep.model_failure,
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
            "propagated": ex.labels.propagated,
            "propagated_from": ex.labels.propagated_from,
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
        # Over all episodes (benign included): the model malformed calls, independent of any attack.
        "malformed_call_count": sum(len(e.malformed_calls) for e in episodes),
        "model_failure_episodes": sum(1 for e in episodes if e.model_failure),
        "by_operator": _group_rates(attacked, lambda e: e.operator.value),
        "by_hop": _group_rates(attacked, lambda e: e.fault_hop or "none"),
    }


def _hardening_blocked(episodes: list[EpisodeOutcome]) -> dict[str, Any]:
    """Tool calls the hardening middleware failed, by reason and by attack operator."""
    by_reason: dict[str, int] = {}
    by_operator: dict[str, int] = {}
    for ep in episodes:
        for rec in ep.records:
            reason = rec.hardening.get("failure_reason")
            if reason:
                by_reason[reason] = by_reason.get(reason, 0) + 1
                op = rec.attack.operator.value
                by_operator[op] = by_operator.get(op, 0) + 1
    return {"total": sum(by_reason.values()), "by_reason": by_reason, "by_operator": by_operator}


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
