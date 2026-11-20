# k-nearest-neighbours baseline

A small CPU monitor that predicts a banking tool call's cause (`none` / `agent` / `adversary`) from
the labels of its nearest training rows. It reuses the features, metrics and embedding cache from
[`../mlp_baseline/`](../mlp_baseline/) and needs only numpy and scikit-learn (plus `transformers`
and `torch` for `knn_embed`).

## Models

| Model | Row representation |
| --- | --- |
| `knn` | TF-IDF of the state text (`--max-features`, default 4096) |
| `knn_embed` | Sentence embedding of the state (`BAAI/bge-small-en-v1.5` from the local Hugging Face cache; nothing is downloaded) |

Both can append the [comparison features](../mlp_baseline/README.md#comparison-features-checkspy)
(`--use-checks`, scaled as in the MLP). Distances are
cosine; the `--neighbors` (default 5) closest training rows vote, each weighted by closeness and by
1 / class frequency so the rare `adversary` class is not outvoted. Probabilities are not calibrated,
and correctness is `P(cause = none)`.

Use `--view middleware` (no privileged fields, [the views](../mlp_baseline/README.md#what-the-monitor-sees-the-state-string-and-the-two-views)) for anything deployable.

## Run

The unified tester calls this for you (`sim/testing/sweep.py run --model knn`). To run one fold:

```bash
python classifiers/knn_baseline/fit_predict.py --model knn --train train.jsonl --validation val.jsonl \
  --test test.jsonl --out runs/fold --use-checks --view middleware
```

It writes `test_predictions.jsonl` (the tester's format) and `results.json` to `--out`.
`--validation` rows are added to the index. kNN is deterministic, so one training seed per fold is
enough.

## Results

On the core tasks (middleware view) kNN found most errors but raised false alarms on 7–10% of correct
calls, so it's a poor blocker. Current numbers: `sim/testing/results/<config>/summary/SUMMARY.md`.

## Layout

| File | What it does |
| --- | --- |
| `fit_predict.py` | Command line for one fold |
| `knn.py` | The model, its features, and the fold's input checks |
| `tests/` | Tests (`python -m pytest -q classifiers/knn_baseline/tests`) |
