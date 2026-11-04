#!/usr/bin/env python3
"""Build Laya's banking train/val/test from first-party agent runs.

    python laya_finetuning/prepare_banking_data.py --runs agent_env/data/runs \\
      --config testing/configs/agent-v4.json --out laya_finetuning/data/agent-combined-v4

Uses the MLP's pipeline (``mlp_baseline``): merge, repair and deduplicate the runs, split once into
train/test, then carve validation out of the train pool for Laya's early stopping. The run list
comes from ``--runs-list``, then ``--config``, then ``mlp_baseline``'s ``COMBINED_RUNS``.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mlp_baseline" / "src"))

from mlp_baseline.agent_runs import COMBINED_RUNS, LAYA_ROWS_FILE, read_runs_config  # noqa: E402
from mlp_baseline.monitor_datasets import combine_agent_runs, export_laya_banking  # noqa: E402


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    runs = _run_list(args)

    missing = [name for name in runs if not (args.runs / name / LAYA_ROWS_FILE).exists()]
    if missing:
        raise SystemExit(f"Missing run data under {args.runs}: {', '.join(missing)}")

    # The intermediate train/test dataset is only a step towards the Laya export.
    with tempfile.TemporaryDirectory(prefix="laya-combine-") as tmp:
        combined = Path(tmp) / "combined"
        combine_agent_runs(args.runs, combined, seed=args.seed, test_fraction=args.test_fraction, runs=runs)
        manifest = export_laya_banking(combined, args.out, seed=args.seed, val_fraction=args.val_fraction)

    splits = {name: {"rows": split["rows"], "label_counts": split["label_counts"]}
              for name, split in manifest["splits"].items()}
    print(json.dumps({"runs": list(runs), "out": str(args.out), "splits": splits}, indent=2))
    return 0


def _run_list(args) -> tuple[str, ...]:
    if args.runs_list:
        return tuple(args.runs_list)
    if args.config:
        return tuple(read_runs_config(args.config)["runs"])
    return COMBINED_RUNS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True, help="Directory of agent experiment runs")
    parser.add_argument("--out", type=Path, required=True, help="Empty output dir for train/val/test JSONL")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--runs-list", nargs="+", default=None,
                        help="Explicit list of run directory names (overrides --config and COMBINED_RUNS)")
    parser.add_argument("--config", type=Path,
                        help="JSON config whose 'runs' list is used (e.g. testing/configs/agent-v4.json)")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
