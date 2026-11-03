# MLP baseline

This directory contains the complete local CPU baseline: independent Python package,
processed datasets with fixed splits, three trained checkpoints, console logs, predictions,
and evaluation reports. It does not require importing or editing `agent_env`.

The main model classifies **benign messages versus prompt injection**. The separate banking
development model predicts tool-call correctness and `none` / `agent` / `adversary` cause.
Public injection labels do not supply banking correctness or failure attribution.

## Install and check

Run from this directory after checking out the branch:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python -m pytest -q tests
mlp-baseline verify
```

Use a fully isolated venv, without `--system-site-packages`. Training uses CPU and four
threads by default. The package has no Laya dependency and does not use a remote GPU.
`verify` checks the release file hashes, split separation, and all 1,178 saved test
probabilities for each of the three exported injection models.

## Use the trained model

The primary checkpoint is seed 42, chosen before comparing test scores:

```bash
mlp-baseline predict --checkpoint models/injection/seed42 \
  --data data/injection-v1/test.jsonl --out runs/predictions.jsonl
```

Each input row needs `state` and `questions`; labels are optional. The supplied files already
use that format. Load the whole checkpoint directory: inference needs its weights, vocabulary,
IDF values, architecture, and validation calibration together.

## Reproduce training

All processed data is committed, so these commands need no dataset download:

```bash
mlp-baseline train-injection --data data/injection-v1 \
  --out runs/seed42 --seed 42 --epochs 40 --threads 4
mlp-baseline train-injection --data data/injection-v1 \
  --out runs/seed43 --seed 43 --epochs 40 --threads 4
mlp-baseline train-injection --data data/injection-v1 \
  --out runs/seed44 --seed 44 --epochs 40 --threads 4
```

Use a new output directory for each experiment. The 40-epoch setting is a maximum; early
stopping uses patience 8, and the lowest validation cross entropy selects the checkpoint.
The archived console logs were captured from these standalone-package reproduction runs.

To rebuild the exact splits from the pinned public sources:

```bash
mlp-baseline prepare-injection --cache .cache/sources \
  --out runs/regenerated-data --download --seed 42 --similarity 0.9
```

Data provenance, revisions, licenses, transformations, and file hashes are in
[`sources/NOTICE.md`](sources/NOTICE.md) and [`data/injection-v1/manifest.json`](data/injection-v1/manifest.json).
The fixed typed-question files can also be used by the teammate training another model
for an evaluation on the same task and split.

## Banking development experiment

```bash
mlp-baseline prepare-banking --run data/banking/triton42-v2/raw \
  --out runs/banking-clean
mlp-baseline train-banking --run data/banking/triton42-v2/raw \
  --out runs/banking-dev --seed 42 --epochs 40
mlp-baseline predict --checkpoint models/banking-dev \
  --data data/banking/triton42-v2/prepared/laya_train.jsonl \
  --out runs/banking-predictions.jsonl
```

The raw synthetic run is copied from the existing repository. Cleanup repairs two known
legacy no-op IO-attack labels and removes 27 exact duplicate rows. It leaves 58 exact unique
inputs; six more differ only by random bookkeeping IDs, leaving 52 cases for grouped splitting.
Unknown label conflicts are rejected. Raw logs and `baseline.json` remain historical.
The legacy `laya_train.jsonl` filename is retained as a data format, not as a model dependency.

The existing simulator is not modified by this folder. The optional
[`patches/simulator-label-fix.patch`](patches/simulator-label-fix.patch) fixes future labeling
of scheduled IO attacks that do not change a valid tool result. Apply it from the repository
root with `git apply mlp_baseline/patches/simulator-label-fix.patch` if that fix has not already
been incorporated. The prepared data bundled here already includes the corrections.

## First-party head-to-head experiment

This experiment evaluates the three-class monitor task (`none` / `agent` / `adversary`,
with correctness as `P(cause=none)`) on **first-party agent runs only**. External datasets
never appear in any evaluation; they may only train the frozen binary base of the hybrid
scenario. Run from the repository root layout (the runs live in `../agent_env/data/runs`):

```bash
mlp-baseline prepare-agent-data --runs ../agent_env/data/runs \
  --out data/agent-combined-v2 --seed 42
mlp-baseline run-agent-comparison --data data/agent-combined-v2 \
  --out reports/agent-combined-v2 --seed 42 --k 5 \
  --injection-checkpoint models/injection/seed42
```

`prepare-agent-data` merges the runs named in `COMBINED_RUNS` (currently 8 runs across three
agent models: `qwen/qwen3.6-35b-a3b`, `api-muse-glimmer-30b`, `api-glm-5.3-flash`), applies
the legacy no-op IO repair per run (18 repairs), deduplicates identical normalized inputs
across runs (519 raw rows become 134 unique examples), and writes a fixed group-disjoint
train/test split with per-run provenance. `run-agent-comparison` then compares two scenarios
with group-stratified cross-validation (k=5, every fold keeps every class):

- **scenario_a_first_party_mlp**: a three-class TF-IDF MLP trained on the first-party train
  pool only; the epoch count is selected by CV and the final model refits on the full pool.
- **scenario_b_hybrid_stacking**: the frozen external injection MLP (seed 42) acts as a
  feature generator; a first-party multinomial logistic regression stacks the calibrated
  injection probability with TF-IDF features to predict the three causes. Stacking lets
  first-party labels teach that a resisted attack is `none`, not `adversary`.

Both scenarios fit temperature calibration on out-of-fold train-pool predictions (never on
test) and are evaluated once on the same 15 held-out test rows, alongside a majority
baseline, with per-source-run breakdowns and error-type attribution accuracy (whether the
predicted cause type was right, overall and when an error was detected). Outputs:
`reports/agent-combined-v2/{results.json,metrics.csv,test_predictions.jsonl}` and trained
artifacts under `reports/agent-combined-v2/models/`.

With 134 unique examples, 2 agent-error test rows, and only 10 independent request groups,
this remains a development comparison, not a benchmark claim. (v1: 75 uniques, 6 runs.)

## Multi-split sweep

Single held-out splits of this data swing a lot. [`../testing/`](../testing/) runs the MLP,
stacking and Laya over many group cross-validation splits and seeds and reports medians and
paired differences. It trains through this package's `fit-predict` command, which runs one
scenario on given files:

```bash
mlp-baseline fit-predict --model mlp --train train.jsonl --test test.jsonl \
  --out runs/fold --seed 42 --k 5
```

`prepare-agent-data --config ../testing/configs/agent-v4.json` takes the run list from that
config instead of `COMBINED_RUNS`.

## Contents

| Path | Contents |
| --- | --- |
| `src/mlp_baseline/` | Standalone preprocessing, training, calibration, metrics, inference, and verification |
| `tests/` | Focused regression tests, including both two-class and three-class MLPs |
| `data/injection-v1/` | 8,893 train / 1,064 validation / 1,178 test cases and separate ToolFault diagnostics |
| `data/banking/triton42-v2/` | Original synthetic run, repaired data, and provenance |
| `data/agent-combined-v2/` | Combined first-party runs (8 runs, 3 agent models): 134 unique examples, fixed train/test split, provenance |
| `data/agent-combined-v1/` | Superseded combine (6 runs incl. deleted `run_1`): 75 unique examples |
| `models/injection/seed{42,43,44}/` | Trained weights and complete inference metadata |
| `models/banking-dev/` | Separate small-data banking checkpoint |
| `logs/` | Captured training output and structured epoch histories |
| `predictions/` | Per-example held-out predictions for every seed |
| `reports/` | Aggregate metrics, per-source results, audits, and packaging verification |
| `REPORT.md` | Baseline method, results, and limitations |
| `RELEASE.json` | Integrity manifest for the published files |

See [REPORT.md](REPORT.md) for the reported results. False alarms remain material, and the
banking development data is too small for a reliable final model comparison.
