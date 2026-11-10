#!/usr/bin/env python3
"""Index one tester fold's training rows and predict its test rows with a k-nearest-neighbours monitor.

    python knn_baseline/fit_predict.py --model knn --train train.jsonl --validation val.jsonl \\
        --test test.jsonl --out runs/fold --use-checks --view middleware

``knn_embed`` embeds states with a locally cached Hugging Face model (default BAAI/bge-small-en-v1.5).
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mlp_baseline" / "src"))  # checks, common, metrics, embeddings, mlp constants
sys.path.insert(0, str(HERE))

from knn import MODELS, fit_predict  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, help="Optional; these rows are added to the index")
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--view", choices=["full", "middleware"], default="full",
                        help="middleware hides ledger_snapshot and received_args, which a real middleware cannot see")
    parser.add_argument("--use-checks", action="store_true", help="Append the comparison features")
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=4096, help="TF-IDF vocabulary size (knn only)")
    parser.add_argument("--embedding", help="Hugging Face model id for knn_embed (default BAAI/bge-small-en-v1.5)")
    parser.add_argument("--seed", type=int, default=42, help="Accepted for the tester; kNN is deterministic")
    args = parser.parse_args(argv)

    result = fit_predict(args.train, args.test, args.out, model=args.model, view=args.view,
                         use_checks=args.use_checks, neighbors=args.neighbors, max_features=args.max_features,
                         embedding=args.embedding, validation=args.validation)
    print(json.dumps({"model": result["model"], "test_rows": result["test_rows"],
                      "predictions": str(args.out / "test_predictions.jsonl")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
