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

## Contents

| Path | Contents |
| --- | --- |
| `src/mlp_baseline/` | Standalone preprocessing, training, calibration, metrics, inference, and verification |
| `tests/` | Focused regression tests, including both two-class and three-class MLPs |
| `data/injection-v1/` | 8,893 train / 1,064 validation / 1,178 test cases and separate ToolFault diagnostics |
| `data/banking/triton42-v2/` | Original synthetic run, repaired data, and provenance |
| `models/injection/seed{42,43,44}/` | Trained weights and complete inference metadata |
| `models/banking-dev/` | Separate small-data banking checkpoint |
| `logs/` | Captured training output and structured epoch histories |
| `predictions/` | Per-example held-out predictions for every seed |
| `reports/` | Aggregate metrics, per-source results, audits, and packaging verification |
| `REPORT.md` | Baseline method, results, and limitations |
| `RELEASE.json` | Integrity manifest for the published files |

See [REPORT.md](REPORT.md) for the reported results. False alarms remain material, and the
banking development data is too small for a reliable final model comparison.
