# Unified tester

With so little first-party data, one train/test split decides a lot of the score. This folder
evaluates the banking monitors — rules, logistic regression, kNN, MLPs, the stacking model and Laya — over many
cross-validation splits and reports the distribution, not a single number.

The tester **does not train anything itself**. For each fold it calls the model's own command:

| Model | Command it calls | Owned by |
| --- | --- | --- |
| `mlp`, `mlp_embed`, `stacking` | `mlp-baseline fit-predict --model <name> …` | `mlp_baseline/` |
| `rules`, `rules_plus` | `mlp-baseline fit-predict --model <name> …` | `mlp_baseline/` |
| `logreg_checks`, `logreg_memory` | `logreg_baseline/fit_predict.py --model <name> …` | `logreg_baseline/` |
| `knn`, `knn_embed` | `knn_baseline/fit_predict.py --model <name> …` | `knn_baseline/` |
| `laya` | `laya_finetuning/train_laya.py --task banking …` | `laya_finetuning/` |

Every model gets the same files and is scored by the same `mlp_baseline.metrics.evaluate`.

## Code layout

| File | What it does |
| --- | --- |
| `sweep.py` | Command-line runner: `splits`, `run`, `summarize` |
| `sweep_helpers/config.py` | Loads the config; the `Sweep` object knows every result path |
| `sweep_helpers/splits.py` | Pools the runs and writes each fold's train/val/test files |
| `sweep_helpers/models.py` | Builds and runs each model's command, reads its predictions back |
| `sweep_helpers/summary.py` | Scores predictions and writes the per-fold, pooled and paired tables |

## Usage

Run from the repository root with the `mlp_baseline` environment, plus `testing/requirements.txt` (it supplies the merge and
scoring helpers):

```bash
python testing/sweep.py splits    --config testing/configs/agent-v4.json
python testing/sweep.py run       --config testing/configs/agent-v4.json --model mlp
python testing/sweep.py run       --config testing/configs/agent-v4.json --model stacking
python testing/sweep.py run       --config testing/configs/agent-v4.json --model laya
python testing/sweep.py summarize --config testing/configs/agent-v4.json
```

- `--model all` runs every model in the config.
- `--split-seeds`, `--folds` and `--training-seeds` run part of the sweep, e.g. a first Laya check.
- `--dry-run` prints the commands without running them.
- Finished units are skipped, so an interrupted sweep continues where it stopped.
- `summarize` uses whatever has finished; a partial Laya sweep is fine.

## Config

`configs/agent-v4.json` names the agent runs to pool and how to run each model. Paths are
relative to the repository root. To test a different pool, copy the file and change `name` and
`runs`; results go to `results/<name>/`.

| Key | Meaning |
| --- | --- |
| `runs_dir`, `runs` | Agent runs to merge (same repair + dedup as `prepare-agent-data`) |
| `split.seeds` | One distinct k-fold assignment of the request groups per seed |
| `split.k` | Folds per split; every group is tested once per split seed |
| `split.max_test_fraction` | Rejects assignments with a test fold larger than this share of rows |
| `split.laya_val_fraction` | Training groups held out as Laya's validation set |
| `training_seeds` | Training seeds run on every split (a model can override this) |
| `models.<name>.python` | Optional interpreter override; defaults to the Python running the sweep |
| `models.<name>.args` | Extra flags passed to that model's command, e.g. `--use-checks`, `--view middleware` |

The same file works with `mlp-baseline prepare-agent-data --config …` and
`laya_finetuning/prepare_banking_data.py --config …`, so all tools can share one run list.

The tester refuses to reuse a results folder whose runs or split settings changed, and refuses
to mix runs of one model made with different model settings.

## How splits work

1. Merge the configured runs and group rows by user request.
2. For each split seed, shuffle the groups into `k` folds. Keep the assignment only if every
   test fold and every training part has all three causes, no test fold is too large, and the
   assignment differs from earlier split seeds.
3. In each fold, hold out some training groups as `val.jsonl`. Laya uses it for early stopping;
   the MLP and stacking add it back to training and use inner cross-validation instead.

Run counts are `split seeds × k × training seeds` per model: 75 for the MLP and stacking and 25
for Laya with the default config.

## Outputs

```text
results/<name>/
  config.json, splits.json                      config copy, pool and fold assignments
  data/p<split>/f<fold>/{train,val,test}.jsonl  the files every model receives
  runs/<model>/p<split>_f<fold>_t<seed>/        predictions.jsonl, metrics.json, run_config.json,
                                                log.txt, native/ (the model's own output)
  summary/
    results.csv    THE RESULTS: median, mean and std per model and metric
    SUMMARY.md     the same as a table, plus model-vs-model differences
    summary.json   the same plus quartiles and min/max
    raw/           repeats.csv (one row per repeat), per_fold.csv, paired.csv
```

Read `results.csv` (or the table in `SUMMARY.md`). Each value is the median or mean over all
**repeats**. A repeat is one split seed trained with one training seed: its k fold predictions
together cover every row once, so they are scored as one set. The files in `raw/` hold the
individual repeats and folds behind those numbers.

## Caveats

- Laya trains on fewer groups than the MLP and stacking, because it needs a fixed validation set.
- Laya's banking validation sets have under 20 rows, so it stays uncalibrated (temperature 1).
- Pooled scores combine k separately trained fold models.

## Tests

```bash
python -m pytest -q testing/tests
```

The tests replace model training with `tests/fake_model.py`.
