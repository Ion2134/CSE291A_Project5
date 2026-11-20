# Recovery

Everything about undoing bad tool calls while an agent runs, in two modules:

| Module | What it does |
| --- | --- |
| `recovery_classifiers.py` | The banking monitors as plug-in **classifiers** the simulator asks about every tool call, plus the majority vote (`decide`) |
| `recovery.py` | The **rules** for when a call is rolled back (`RecoveryPolicy`), and the **rollback** of the recorded state (`rollback`) |
| `interpret_results.py` | Breaks a recovery run down: outcomes, false positives, why each rollback happened ([commands](COMMANDS.md)) |
| `FINDINGS.md` | Known classifier and recovery issues seen in runs, for reports |

Two packages it relies on live inside this folder: [`state_db/`](state_db/README.md) (the recorded
state) and [`mcp_hardening/`](mcp_hardening/README.md) (request/response signing and telemetry).

`sim/agent_env` uses them with `--recovery-classifiers` (score and vote) and `--recovery` (act on it). The
state a rollback restores lives in the [state DB](state_db/README.md). `recovery.py` uses only the
standard library; `recovery_classifiers.py` needs [`mlp_baseline`](../classifiers/mlp_baseline/) and finds
`classifiers/logreg_baseline/` and `classifiers/knn_baseline/` from the repo checkout.

## Setup

```bash
python3.12 -m venv recovery/.venv      # an arm64 Python: torch has no Intel-Mac builds past 2.2
recovery/.venv/bin/pip install -e classifiers/mlp_baseline -e recovery/state_db -e "recovery[dev]"
sim/agent_env/.venv/bin/pip install -e classifiers/mlp_baseline -e recovery     # so the simulator can import it
```

Tests: `cd recovery && .venv/bin/python -m pytest -q`.

## The classifiers (`recovery_classifiers.py`)

Every classifier has the same two methods (dependency injection: the simulator only knows this shape):

```python
fit(train_rows) -> self           # no-op for the ones that need no training
predict(rows)   -> (n, 3) array    # cause probabilities: none, agent, adversary
```

Rows are the monitor format used everywhere else (`{"state", "expected", "questions"}`, as in
`laya_train.jsonl`). All classifiers except `oracle` see the **middleware view** (no `ledger_snapshot`,
no `received_args`) unless their spec sets `"view": "full"`.

| Name | Needs | Wraps |
| --- | --- | --- |
| `oracle` | nothing | the true label (a perfect classifier: the recovery upper bound) |
| `rules` | nothing | `mlp_baseline.checks.rule_cause` |
| `rules_plus` | `train` | `mlp_baseline.variants.rule_cause_plus` |
| `logreg_checks`, `logreg_memory` | `train` | `classifiers/logreg_baseline/logreg.py` (`fit_logreg`) |
| `knn`, `knn_embed` | `train` | `classifiers/knn_baseline/knn.py` (`fit_knn`); `knn_embed` also needs `transformers` |
| `mlp`, `mlp_embed`, `stacking` | `checkpoint` | a model folder saved by `mlp-baseline` (`load_mlp`, `load_stacker`) |

`laya` isn't wrapped yet (it needs its own environment). To add a classifier, write a class with
`fit`/`predict` in `recovery_classifiers.py` and add it to `CLASSIFIERS`.

### Specs

A run lists them with `--recovery-classifiers` (or `"recovery_classifiers"` in a batch config) as a
JSON list. Each entry is a name, or a dict with the name and its inputs:
- Paths are relative to the repo root.
- Other keys are passed to the classifier, e.g. `"injection_detector": "char_ngram"` for logreg, or
  `"neighbors": 7` for kNN.
- `"label"` lets one classifier appear twice under different names.

```json
[
  "oracle",
  "rules",
  {"name": "logreg_checks", "train": "sim/testing/results/agent-v8-logreg/data/p42/f0/train.jsonl"},
  {"name": "logreg_checks", "label": "logreg_ngram", "train": "sim/testing/results/agent-v8-logreg/data/p42/f0/train.jsonl",
   "injection_detector": "char_ngram"}
]
```

Classifiers are built and fitted once per run, before the first episode.

### The vote

A classifier **flags** a call when its most likely cause is `agent` or `adversary`. The vote says
"recover" when **more than half** of the classifiers flag the call (`decide`). With an empty list,
no classifiers are called.

Classifiers aren't asked about calls [MCP hardening](mcp_hardening/README.md) already rejected.
Those are an error with empty output, which they never trained on, so `classifier_outputs` is `null`
for them.

## When a call is rolled back (`recovery.py`)

`RecoveryPolicy.decide(hardening_failure_stage, classifier_vote)` applies these rules in order:

| Situation | Rolled back? | `reason` |
| --- | --- | --- |
| Hardening rejected the call **before** the tool ran (`bad_request_sig`, replay, stale timestamp) | no: nothing changed | `blocked_before_tool` |
| Hardening rejected the **response** after the tool ran (`bad_response_sig`) | yes: the tool's effect is undone | `response_sig_failed` |
| The classifiers' vote says recover | yes | `classifiers` |
| It doesn't | no | `not_flagged` |
| No classifiers configured | no | `no_classifiers` |

There's one policy per episode, with a cap: after `max_recoveries` rollbacks (default 2), a call that
would be rolled back isn't (`cap_reached`). This stops an agent that keeps repeating a bad call from
looping forever.

Without `--recovery`, the vote is only recorded and nothing is rolled back.

## What a rollback does

`rollback(store, episode_id, to_seq)` calls the state store's `delete_after(episode_id, to_seq)` and
returns the ledger and conversation as of `to_seq`. In `agent_env`:

1. `to_seq` is the step right before the flagged call ran: the assistant message that made it.
2. The ledger is restored in place from that snapshot, which undoes the transfer or side effect.
3. The call is removed from the agent's context, so later calls' state doesn't show it.
4. The agent gets `BLOCKED_REPLY` (`{"error": "blocked by the monitor; state restored to before this call"}`)
   instead of the output, and carries on within its step budget.

## Where it shows up

| Where | What |
| --- | --- |
| `examples.jsonl` → `meta.classifier_outputs` | `{label: {none, agent, adversary}}` for that call |
| `examples.jsonl` → `meta.recovery` | the vote `{recover, flagged_by}`; with `--recovery` also `reason`, `rolled_back`, `to_seq`. Rolled-back calls stay here (they're real training data). |
| `state.db` → `tool_calls` | `classifier_outputs`, `recover`, `call_id`; rolled-back calls are deleted (it holds the current state) |
| telemetry | the span is kept (it's the history), with `discovery_sim.recovery.rolled_back` and `.reason` |
| `baseline.json` → `recoveries` | rollbacks in total, and decisions counted by reason (with `--recovery`) |
| `run_manifest.json` | `recovery_classifiers` (the specs), `recovery`, `max_recoveries` |
| `task_correct` | judged on the calls that stand: rolled-back calls' labels don't count, and the final ledger must satisfy the task |

To join a DB row to its telemetry span, use `(episode_id, call_id)`; `call_id` is the span's `gen_ai.tool.call.id`.
