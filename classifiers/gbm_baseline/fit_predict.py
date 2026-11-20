#!/usr/bin/env python3
"""Train one gradient-boosted-trees monitor on a train file and predict a test file (one tester fold).

    python classifiers/gbm_baseline/fit_predict.py --model gbm_checks --train train.jsonl \\
        --validation val.jsonl --test test.jsonl --out runs/fold --view middleware

Same inputs, outputs and flags as ``logreg_baseline/fit_predict.py``; reuses its features.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mlp_baseline" / "src"))  # checks, common, metrics
sys.path.insert(0, str(HERE.parent / "logreg_baseline"))       # the shared features
sys.path.insert(0, str(HERE))

from injection import DETECTORS  # noqa: E402
from gbm import MODELS, fit_predict  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, help="Optional; these rows are added to training")
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--view", choices=["full", "middleware"], default="full",
                        help="middleware hides ledger_snapshot and received_args, which a real middleware cannot see")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--injection-detector", choices=sorted(DETECTORS), default="keyword",
                        help="What fills the output_injection_text feature (see logreg_baseline/injection.py)")
    args = parser.parse_args(argv)

    result = fit_predict(args.train, args.test, args.out, model=args.model, view=args.view, seed=args.seed,
                         validation=args.validation, injection_detector=args.injection_detector)
    print(json.dumps({"model": result["model"], "test_rows": result["test_rows"],
                      "predictions": str(args.out / "test_predictions.jsonl")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
