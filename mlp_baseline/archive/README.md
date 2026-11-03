# Archive

Earlier experiments and one-off notes. Nothing here is read by the unified tester
(`../../testing/`) or by Laya; the files are kept so past results stay reproducible. Paths
recorded *inside* these JSON files refer to the folder layout at the time (`data/…`,
`models/…`, `reports/…`), before the move into `archive/`.

## `agent-comparison/v1` … `v4`

The single-split banking-monitor comparison that preceded the unified tester: scenario A (the
first-party MLP) vs scenario B (stacking on the injection detector), each trained once on a
fixed train pool and scored on one held-out split. Results swung a lot between versions because
each used a different split, which is why the tester now reports medians over many splits.

| Version | Runs merged | Unique rows | Test rows | Notes |
| --- | ---: | ---: | ---: | --- |
| v1 | 6 | 75 | 15 | Includes the since-deleted `run_1` |
| v2 | 8 | 134 | 25 | |
| v3 | 9 | 149 | 27 | |
| v4 | 10 | 169 | 33 | Data built with `--seed 500`; Laya's banking split (`laya_finetuning/data/agent-combined-v4`) was frozen from it |

Each version has `data/` (train/test split, provenance, manifest) and `report/` (`results.json`,
`metrics.csv`, `test_predictions.jsonl`, and the trained fold and final models). To rerun v4
from `mlp_baseline/`:

```bash
mlp-baseline run-agent-comparison --data archive/agent-comparison/v4/data \
  --out runs/comparison-v4 --seed 42 --k 5
```

## `banking-dev/`

The first banking experiment: one simulator run (`triton42-v2`, copied into `data/raw/`),
repaired and deduplicated to 52 cases (`data/prepared/`), split by request group, and one
three-class MLP (`model/`, `report/`, `train.log`). Development-only: 11 test rows. The tests use
`data/raw/` as a real-run fixture. To rerun:

```bash
mlp-baseline prepare-banking --run archive/banking-dev/data/raw --out runs/banking-clean
mlp-baseline train-banking --run archive/banking-dev/data/raw --out runs/banking-dev --seed 42 --epochs 40
```

## Notes

- `dataset-audit/`: the audit of external datasets considered for training (`assessment.json`,
  `audit.json`), including why AFTraj was not used.
- `packaging-verification.json`: what was checked when the package was first released (test
  counts at that time).
- `simulator-label-fix.patch`: fixes the simulator's labeling of scheduled IO-mismatch attacks
  that cannot change a correct result. Not yet applied to `agent_env`;
  `prepare-agent-data` repairs affected rows instead. Apply from the repository root with
  `git apply mlp_baseline/archive/simulator-label-fix.patch`.
