# Logistic-regression baseline

A small CPU monitor: multinomial logistic regression on the banking monitor's **comparison
features**. It is a learned version of the hand-written `rules` cascade, so it reaches the same
accuracy with real probabilities and weights you can read. It needs only numpy and scikit-learn,
and reuses the features and metrics from [`../mlp_baseline/`](../mlp_baseline/) (`checks.py`,
`metrics.py`, `common.py`).

## Models

| Model | Features | Notes |
| --- | --- | --- |
| `logreg_checks` | The 12 comparison flags (call vs mandate, args vs call, output vs args and ledger, tool kind) | The one to headline. No dependence on simulator constants. |
| `logreg_memory` | The 12 flags plus `balance_unseen` and `empty_transfer_output` | Read as an upper bound: the extra checks were found by inspecting the folds, and the balance check relies on the simulator reusing the same seeded balances. |

The feature definitions are in [`mlp_baseline/README.md`](../mlp_baseline/README.md#comparison-features-checkspy).
Classes are weighted equally in the loss (`class_weight="balanced"`), `C=1`, and the weights learned
for each cause are saved in `results.json` under `details.weights`.

`logreg_memory`'s two extra features are built from the training rows only. For the training rows
themselves, each row's memory leaves out its own request group, so a row is never looked up in its
own memory.

## Views

`--view middleware` removes `ledger_snapshot` and `received_args` before the features are computed
(see [the views](../mlp_baseline/README.md#what-the-monitor-sees-the-state-string-and-the-two-views));
`--view full` keeps them. Use `middleware` for anything you want to deploy.

## Run

The unified tester calls this for you (`testing/sweep.py run --model logreg_checks`). To run one fold:

```bash
python logreg_baseline/fit_predict.py --model logreg_checks \
  --train train.jsonl --validation val.jsonl --test test.jsonl --out runs/fold --view middleware
```

It writes `test_predictions.jsonl` (the tester's format) and `results.json` (settings, learned
weights and test metrics) to `--out`. `--validation` rows are added to training. It is deterministic,
so one training seed per fold is enough.

## Layout

| File | What it does |
| --- | --- |
| `fit_predict.py` | Command line for one fold |
| `logreg.py` | The model, the memory features and the fold's input checks |
| `tests/` | Tests (`python -m pytest -q logreg_baseline/tests`) |
