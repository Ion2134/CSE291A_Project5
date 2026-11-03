#!/usr/bin/env python3
"""Combine first-party agent runs into Laya banking train/val/test.

Same allowlist pattern as ``mlp_baseline``: edit ``COMBINED_RUNS`` to name the
experiment directories under ``agent_env/data/runs``, then:

    python laya_finetuning/prepare_banking_data.py \\
      --runs agent_env/data/runs \\
      --out laya_finetuning/data/agent-combined-v4

Merges/repairs/dedups via the MLP combine pipeline, freezes the MLP test split,
and carves validation from the train pool for Laya early-stopping/calibration.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

# Same experiment dirs as mlp_baseline.agent_data.COMBINED_RUNS — edit both when adding runs.
COMBINED_RUNS = (
    "agent-qwen36a3b",
    "agent42",
    "agent42-v2",
    "triton42",
    "triton42-v2",
    "triton44",
    "triton45",
    "triton46",
    "triton47",
    "triton48",
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True, help="Directory of agent experiment runs")
    parser.add_argument("--out", type=Path, required=True, help="Empty output dir for train/val/test JSONL")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--runs-list",
        nargs="+",
        default=None,
        help="Override COMBINED_RUNS with an explicit list of run directory names",
    )
    args = parser.parse_args(argv)

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "mlp_baseline" / "src"))
    from mlp_baseline.agent_data import COMBINED_RUNS as MLP_RUNS, combine_agent_runs, export_laya_banking

    runs = tuple(args.runs_list) if args.runs_list else COMBINED_RUNS
    if runs != MLP_RUNS and args.runs_list is None:
        print(
            "warning: laya COMBINED_RUNS differs from mlp_baseline.agent_data.COMBINED_RUNS; "
            "keep them in sync unless you intend a different pool",
            file=sys.stderr,
        )

    missing = [name for name in runs if not (args.runs / name / "laya_train.jsonl").exists()]
    if missing:
        raise SystemExit(f"Missing run data under {args.runs}: {', '.join(missing)}")

    with tempfile.TemporaryDirectory(prefix="laya-combine-") as tmp:
        combined = Path(tmp) / "combined"
        combine_agent_runs(
            args.runs, combined, seed=args.seed, test_fraction=args.test_fraction, runs=runs,
        )
        manifest = export_laya_banking(
            combined, args.out, seed=args.seed, val_fraction=args.val_fraction,
        )

    print(json.dumps({
        "runs": list(runs),
        "out": str(args.out),
        "splits": {
            name: {"rows": split["rows"], "label_counts": split["label_counts"]}
            for name, split in manifest["splits"].items()
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
