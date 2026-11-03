#!/usr/bin/env python3
"""Unified tester: compare the banking monitors over many cross-validation splits.

The tester owns the splits and the scoring; it never trains anything itself. For every
(split seed, fold, training seed) it calls each model's own training command on the same
files and scores all predictions with the same ``mlp_baseline.metrics.evaluate``.

    python testing/sweep.py splits    --config testing/configs/agent-v4.json
    python testing/sweep.py run       --config testing/configs/agent-v4.json --model mlp
    python testing/sweep.py summarize --config testing/configs/agent-v4.json

Results go to ``testing/results/<config name>/``. See ``testing/README.md``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sweep_helpers.config import MODELS, Sweep, load_config
from sweep_helpers.models import run_models
from sweep_helpers.splits import make_splits
from sweep_helpers.summary import summarize


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("splits", help="Pool the runs and write the fold files")
    run = commands.add_parser("run", help="Train and score one model (or all) on every fold")
    run.add_argument("--model", choices=[*MODELS, "all"], required=True)
    run.add_argument("--split-seeds", type=int, nargs="+", help="Only these split seeds")
    run.add_argument("--folds", type=int, nargs="+", help="Only these folds")
    run.add_argument("--training-seeds", type=int, nargs="+", help="Only these training seeds")
    run.add_argument("--dry-run", action="store_true", help="Print the commands without running them")
    commands.add_parser("summarize", help="Write the summary tables from finished runs")
    for command in commands.choices.values():
        command.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)


    sweep = Sweep(load_config(args.config))
    manifest = make_splits(sweep)  # Every command needs the folds; they are only built once.


    if args.command == "splits":
        print(json.dumps(manifest["pool"], indent=2))
        print(f"Wrote folds to {sweep.out / 'data'}")

    elif args.command == "run":
        models = [m for m in MODELS if m in sweep.models] if args.model == "all" else [args.model]
        run_models(sweep, models, manifest, split_seeds=args.split_seeds, folds=args.folds,
                   training_seeds=args.training_seeds, dry_run=args.dry_run)
        
    else:
        summarize(sweep, manifest)
        print(f"Wrote {sweep.out / 'summary'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
