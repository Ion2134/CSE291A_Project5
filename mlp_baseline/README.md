# MLP baseline

A standalone CPU package with two jobs:

1. **The injection detector**: an MLP that classifies a message as **benign** or **prompt
   injection**, trained on public datasets. Its checkpoint is also the frozen base of the
   `stacking` model.
2. **The banking monitor MLPs** (`mlp` and `stacking`): they predict whether a banking tool call's
   output is correct and, if not, the cause (`none` / `agent` / `adversary`). These are trained
   and compared against Laya by the unified tester in [`../testing/`](../testing/), which calls
   this package's `fit-predict` command.

It does not import or edit `agent_env`, and has no Laya dependency.

## Layout

| Path | Contents |
| --- | --- |
| `src/mlp_baseline/` | The package (see [Code](#code) below) |
| `tests/` | Regression tests |
| `data/injection/` | Injection dataset: 8,893 train / 1,064 val / 1,178 test messages, plus ToolFault diagnostics; also Laya's `injection` task data |
| `experiments/injection/` | The trained injection detector: `summary.json` and `metrics.csv` across seeds, and per seed `seedN/{model,predictions,results.json,train.log,history.csv}` |
| `sources/` | Dataset cards, licenses and attribution for the public data |
| `archive/` | Earlier experiments and one-off notes, kept for the record ([`archive/README.md`](archive/README.md)) |
| `REPORT.md` | Method, results and limitations of the injection detector |
| `RELEASE.json` | File hashes checked by `mlp-baseline verify` |

## Install and check

Run from this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python -m pytest -q tests
mlp-baseline verify
```

Use a fully isolated venv, without `--system-site-packages`. Training uses the CPU and four
threads by default. `verify` checks the file hashes in `RELEASE.json`, split separation, and
all 1,178 saved test probabilities for each of the three injection checkpoints.

## Injection detector

The primary checkpoint is seed 42, chosen before comparing test scores:

```bash
mlp-baseline predict --checkpoint experiments/injection/seed42/model \
  --data data/injection/test.jsonl --out runs/predictions.jsonl
```

Each input row needs `state` and `questions`; labels are optional. Load the whole `model/`
directory: inference needs its weights, vocabulary, IDF values, architecture and validation
calibration together.

To retrain (all processed data is committed, so no download is needed):

```bash
mlp-baseline train-injection --data data/injection --out runs/seed42 --seed 42 --epochs 40 --threads 4
```

`scripts/reproduce.sh` runs seeds 42, 43 and 44. The 40 epochs are a maximum: early stopping
uses patience 8, and the lowest validation cross entropy selects the checkpoint. To rebuild the
exact splits from the pinned public sources:

```bash
mlp-baseline prepare-injection --cache .cache/sources \
  --out runs/regenerated-data --download --seed 42 --similarity 0.9
```

Provenance, revisions, licenses and file hashes are in [`sources/NOTICE.md`](sources/NOTICE.md)
and [`data/injection/manifest.json`](data/injection/manifest.json).

## Banking monitor

Results come from the unified tester ([`../testing/README.md`](../testing/README.md)), which
builds cross-validation folds from the agent runs and calls, for each fold:

```bash
mlp-baseline fit-predict --model mlp --train train.jsonl --validation val.jsonl --test test.jsonl \
  --out runs/fold --seed 42 --k 5
```

- **`mlp`**: a three-class TF-IDF MLP trained on first-party rows only. Inner group
  cross-validation picks the epoch count and fits temperature calibration on out-of-fold
  predictions; the final model is refit on all training rows.
- **`stacking`**: the frozen injection detector (`experiments/injection/seed42/model`) supplies
  P(injection) as a feature; a logistic regression on that plus TF-IDF features predicts the
  cause. It lets first-party labels teach that a resisted attack is `none`, not `adversary`.

`--validation` rows are added to training, because the inner cross-validation already holds
data out. Correctness is `P(cause = none)`.

Two commands build datasets from agent runs (`agent_env/data/runs/<name>/`), taking the run
list from a tester config:

```bash
mlp-baseline prepare-agent-data --runs ../agent_env/data/runs \
  --config ../testing/configs/agent-v4.json --out runs/combined --seed 42
mlp-baseline export-laya-banking --data runs/combined --out runs/laya-banking
```

`prepare-agent-data` checks each run's `laya_train.jsonl` against its `examples.jsonl`, repairs
a known legacy labeling bug (an IO-mismatch attack that could not change a correct result was
labeled an adversary error), merges and deduplicates rows across runs, and writes a
group-disjoint train/test split. `export-laya-banking` writes the same split in Laya's
train/val/test format.

## Code

| Module | What it does |
| --- | --- |
| `common.py` | Cause labels, row identity (`input_id`) and request grouping, file helpers, `RunRecord` |
| `mlp.py` | The TF-IDF MLP and `TrainSettings`: train, save, load |
| `metrics.py` | Banking-monitor metrics, temperature calibration, majority baseline |
| `splits.py` | Group-disjoint stratified partitions and cross-validation folds |
| `agent_runs.py` | Read agent runs, repair the legacy label bug, merge them into unique rows |
| `monitor_datasets.py` | Merged runs → train/test dataset (`prepare-agent-data`), checks, Laya export |
| `monitor_training.py` | The `mlp` and `stacking` monitors and `fit-predict` (one tester fold) |
| `stacking.py` | The stacked monitor on the frozen injection detector |
| `comparison.py` | The archived single-split comparison report (`run-agent-comparison`) |
| `injection_data.py` | Builds the injection dataset from the pinned public sources |
| `injection_model.py` | Loads, runs and scores the injection detector |
| `injection_training.py` | Trains and reports the injection detector (`train-injection`) |
| `banking_dev.py` | The archived banking-dev experiment (`prepare-banking`, `train-banking`) |
| `verify.py`, `cli.py` | `mlp-baseline verify` and the command line |

See [REPORT.md](REPORT.md) for the injection detector's results; false alarms remain material.
