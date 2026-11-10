# Plan: put the monitor in the loop so the agent recovers (reduce the env's silent errors)

## Context
Offline, the comparison-feature MLP now matches the hand rule (v6: cause macro-F1 0.98, from 0.68). But that
score uses fields a real middleware may not have (`ledger_snapshot`, `received_args`), and it never changes what
the agent does. The end goal is to run the monitor in the middleware (between the agent and the tool client) and
have the agent recover.

**Success = the environment's own numbers.** Each run's `agent_env/data/runs/*/baseline.json` is the no-monitor
baseline for the env + its agent: `task_correct_rate` 0.15–0.82 (0.23 on `triton49-agentheavy`, 0.15 on
`oracle49-agentheavy`), `silent_wrong_rate` up to 0.84, per-operator and per-hop breakdowns. We re-run the same
seeds with the monitor on and move those numbers. Decisions made: middleware sits between agent and tool client;
on a flag the call/output is blocked with an error message and the agent retries.

## What the sim code tells us (why the design below)
- `orchestrator.run_episode` feeds the tool result straight back to the agent. The fault fires on one tool round
  (`fault_call_index`), so a retry after a block is a clean call. The `llm_*` mutation is applied inside
  `harness.execute_tool_call`, after the agent spoke, and the agent's transcript keeps its original args. So
  "agent errors" are mutated calls the agent never sees.
- Two interception points are needed:
  - **Pre-call gate** (on the call after `apply_llm_attack`, before the A2A send): catches `llm_wrong_account` /
    `llm_wrong_amount`, the bulk of errors (177 of 197 `agent` rows), using only `trusted` + `model_call`. For
    `initiate_transfer` this is the only point where harm is prevented, since a wrong transfer already moved money.
  - **Post-call filter** (tool output before it reaches the agent): catches `mcp_*` (extra `_*` keys, injected
    text, output not matching the call) and `a2a_call_intercepted` (output reflects rewritten args, so output
    differs from `model_call`). If a transfer already executed, void it with the existing `cancel_payment` tool.
- `task_correct` is `all(ex.labels.output_correct)`, so a recovered episode still scores wrong. The A/B must
  count only executed, un-voided calls (see Step 4), otherwise recovery is invisible.

## Steps

**1. Deployable view of the state (`mlp_baseline/src/mlp_baseline/checks.py`).**
Add `middleware_view(state)` that drops `ledger_snapshot` and `received_args`, and compute the output checks
against `model_call` instead of `received_args`; balance checks use consistency with prior reads in `prior_calls`
instead of the ledger. Add a `view` option to `comparison_features` / `rule_cause`. New `agent-v7.json` sweep
reports mlp / stacking / rules under this view, to see which operators survive without privileged fields.
Expect agent and injection errors to hold; `mcp_io_mismatch` (wrong balance, no ledger) is the likely loser.

**2. Pre-call and post-call models.**
Derive pre-call rows from existing rows by dropping the `output:` line (label `agent` vs not, since `adversary`
rows have a correct call). No new data generation. Train one MLP per phase via the existing `fit_predict` path;
add a `train-monitor` CLI command that fits on the whole pool and saves a checkpoint (reuse
`banking_dev.load_banking_mlp` / `MLPPredictor`; keep `use_checks` in `model_config.json`). Calibrate thresholds
for a low false-block rate on benign calls, since a blocked benign call costs a retry.

**3. Monitor hook in the sim (add-only; default off, data generation unchanged).**
- `sim/harness.py`: let `execute_tool_call` take an optional `pre_send(call)` callback, invoked after the
  `llm_*` mutation and before the A2A send; if it blocks, return a record marked `blocked` with no ledger effect.
- `sim/orchestrator.py`: accept an optional `monitor` (protocol with `pre_call` and `post_call` returning
  allow/block + reason). On block, append a tool message `{"error": "blocked: <reason>, retry with the mandate"}`
  and continue the loop (don't consume the fault round, so the fault still fires once). On a post-call flag of a
  completed transfer, call `cancel_payment` on the ledger and return an error to the agent.
- `sim/run_config.py`, `cli.py`: `--monitor-checkpoint`, `--monitor-threshold`. Keep `mlp_baseline` out of
  `agent_env`: the sim only sees a callable, and a thin adapter in `testing/closed_loop/` loads the checkpoint.

**4. Closed-loop A/B (`testing/closed_loop/`).**
Run the same seeds with the monitor off and on, and write a `baseline.json`-compatible report with:
`task_correct_rate`, `silent_wrong_rate`, `hard_fail_rate` by operator and hop; plus blocks, retries, voided
transfers, **false-block rate on benign episodes**, and mean extra steps. Fix the metric: compute task
correctness over executed, un-voided calls and final ledger state (`mandate_satisfied`), recording blocked calls
separately. Develop on `--llm-backend oracle` (deterministic, on-mandate retries, so any remaining failure is the
monitor's), then confirm on `triton` (needs keys in `agent_env/.env`).

**5. Close the gap iteratively.** Mine missed/false-blocked cases from the A/B. Retrain on closed-loop
trajectories too, because post-block states (error messages, retries) are not in the current data. Then the
earlier data-hole fixes: more adversary rows via `--operator-weights`, drop or relabel the ~8 unlearnable
rows (`_repair_and_deduplicate` in `agent_runs.py`), and a second real planner model.

## Caveats to keep visible
- The fault fires once per episode, so retry-based recovery is easier here than against a persistent attacker.
- Labels and the A/B come from the same simulator; this shows the loop works, not that it generalizes.
- Laya stays out of scope until the MLP loop works (OOM on this machine).

## Critical files
- `mlp_baseline/src/mlp_baseline/checks.py`, `monitor_training.py`, `cli.py`, `banking_dev.py` (checkpoint loading)
- `agent_env/src/discovery_sim/sim/harness.py`, `orchestrator.py`, `run_config.py`; `agent_env/src/discovery_sim/cli.py`
- new `testing/closed_loop/` (adapter + A/B runner), new `testing/configs/agent-v7.json`
- existing to reuse: `AgentContext.record_call`, `mandate_satisfied`, `label_call`, `cancel_payment` tool, `baseline.json` writer

## Verification
- `python -m pytest -q` for `agent_env/tests`, `mlp_baseline/tests`, `testing/tests` (using `agent_env/.venv`, the only venv with torch);
  add tests for `middleware_view` and for the hook (blocked call leaves the ledger unchanged).
- Offline: `sweep.py run --config agent-v7.json` for mlp/stacking/rules; per-operator recall under the deployable view.
- Closed loop, oracle backend: monitor-off vs monitor-on on seed 49 (`llm_wrong_*` heavy). Target: `task_correct_rate`
  up from 0.15–0.23 toward ~0.9, `silent_wrong_rate` toward 0, benign false-block rate near 0.
- Repeat with `triton` on a second seed set and on mixed operators (seed 47/48 style), reporting the by-operator table.
