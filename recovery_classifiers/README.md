# Recovery classifiers

The banking monitors as plug-in classifiers, so the simulator (`agent_env`) can ask them about every
tool call **while an episode runs**. Each call's outputs are recorded together with a deterministic
decision on whether recovery should run. Recovery itself is not built yet; the decision is only recorded.

## Setup

```bash
python3.12 -m venv recovery_classifiers/.venv     # an arm64 Python: torch has no Intel-Mac builds past 2.2
recovery_classifiers/.venv/bin/pip install -e mlp_baseline -e "recovery_classifiers[dev]"
agent_env/.venv/bin/pip install -e mlp_baseline -e recovery_classifiers      # so the simulator can import it
```

`logreg_baseline/` and `knn_baseline/` are found from the repo checkout. Tests:
`cd recovery_classifiers && .venv/bin/python -m pytest -q`.

## The interface

Every classifier has the same two methods; the simulator only knows this shape:

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
| `logreg_checks`, `logreg_memory` | `train` | `logreg_baseline/logreg.py` (`fit_logreg`) |
| `knn`, `knn_embed` | `train` | `knn_baseline/knn.py` (`fit_knn`); `knn_embed` also needs `transformers` |
| `mlp`, `mlp_embed`, `stacking` | `checkpoint` | a model folder saved by `mlp-baseline` (`load_mlp`, `load_stacker`) |

`laya` is not wrapped yet (it needs its own environment). To add a classifier, write a class with
`fit`/`predict` in `recovery_classifiers.py` and add it to `CLASSIFIERS`.

## Specs

A run lists them with `--recovery-classifiers` (or `"recovery_classifiers"` in a batch config) as a JSON list. Each entry is a name, or a dict with the name and its inputs.
Paths are relative to the repo root. Other keys are passed to the classifier (e.g.
`"injection_detector": "char_ngram"` for logreg, `"neighbors": 7` for kNN). `"label"` lets one
classifier appear twice under different names.

```json
[
  "oracle",
  "rules",
  {"name": "logreg_checks", "train": "testing/results/agent-v8-logreg/data/p42/f0/train.jsonl"},
  {"name": "logreg_checks", "label": "logreg_ngram", "train": "testing/results/agent-v8-logreg/data/p42/f0/train.jsonl",
   "injection_detector": "char_ngram"}
]
```

Classifiers are built and fitted once per run, before the first episode.

## The decision

A classifier **flags** a call when its most likely cause is `agent` or `adversary`. Recovery should run
when **more than half** of the classifiers flag it (`decide`). With an empty list nothing is called.

## Where outputs go

| Where | What |
| --- | --- |
| `examples.jsonl` → `meta.classifier_outputs` | `{label: {none, agent, adversary}}` for that call |
| `examples.jsonl` → `meta.recovery` | `{"recover": bool, "flagged_by": [labels]}` |
| `state.db` → `tool_calls` | the same, as `classifier_outputs` and `recover` (with `--state-db`) |
| `run_manifest.json` → `recovery_classifiers` | the specs the run used |
