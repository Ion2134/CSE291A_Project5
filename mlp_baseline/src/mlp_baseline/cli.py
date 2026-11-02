from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Standalone local MLP baseline")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-injection", help="Regenerate public-data splits from pinned sources")
    prepare.add_argument("--cache", type=Path, default=Path(".cache/dataset-audit"))
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--similarity", type=float, default=0.9)
    prepare.add_argument("--download", action="store_true")
    train = sub.add_parser("train-injection", help="Train the CPU injection-detection MLP")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--max-features", type=int, default=4096)
    train.add_argument("--patience", type=int, default=8)
    train.add_argument("--threads", type=int, default=4)
    bank_prepare = sub.add_parser("prepare-banking", help="Repair known legacy labels and deduplicate a simulator run")
    bank_prepare.add_argument("--run", type=Path, required=True)
    bank_prepare.add_argument("--out", type=Path, required=True)
    bank = sub.add_parser("train-banking", help="Train the separate banking development baseline")
    bank.add_argument("--run", type=Path, required=True)
    bank.add_argument("--out", type=Path, required=True)
    bank.add_argument("--seed", type=int, default=42)
    bank.add_argument("--epochs", type=int, default=40)
    bank.add_argument("--batch-size", type=int, default=16)
    bank.add_argument("--threads", type=int, default=4)
    combine = sub.add_parser("prepare-agent-data", help="Combine first-party agent runs into one pruned dataset")
    combine.add_argument("--runs", type=Path, required=True)
    combine.add_argument("--out", type=Path, required=True)
    combine.add_argument("--seed", type=int, default=42)
    combine.add_argument("--test-fraction", type=float, default=0.2)
    compare = sub.add_parser("run-agent-comparison", help="Head-to-head first-party MLP vs hybrid stacking")
    compare.add_argument("--data", type=Path, required=True)
    compare.add_argument("--out", type=Path, required=True)
    compare.add_argument("--seed", type=int, default=42)
    compare.add_argument("--k", type=int, default=5)
    compare.add_argument("--epochs", type=int, default=40)
    compare.add_argument("--batch-size", type=int, default=16)
    compare.add_argument("--max-features", type=int, default=4096)
    compare.add_argument("--threads", type=int, default=4)
    compare.add_argument("--injection-checkpoint", type=Path, default=Path("models/injection/seed42"))
    compare.add_argument("--verbose", action="store_true")
    predict = sub.add_parser("predict", help="Load an exported checkpoint and predict JSONL examples")
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--data", type=Path, required=True)
    predict.add_argument("--out", type=Path, required=True)
    verify = sub.add_parser("verify", help="Check release hashes, dataset splits, and deployed predictions")
    verify.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    if args.command == "prepare-injection":
        from .injection_data import prepare_injection_data
        report = prepare_injection_data(args.cache, args.out, seed=args.seed,
                                        threshold=args.similarity, download=args.download)
        print(json.dumps(report["splits"], indent=2))
    elif args.command == "train-injection":
        from .injection_mlp import train_injection_mlp
        options = vars(args).copy()
        options.pop("command")
        train_injection_mlp(**options)
    elif args.command == "prepare-banking":
        from .banking_data import prepare_training_data
        print(json.dumps(prepare_training_data(args.run, args.out), indent=2))
    elif args.command == "train-banking":
        from .banking import run_banking
        options = vars(args).copy()
        options.pop("command")
        result = run_banking(**options)
        print(json.dumps({"status": result["status"], "results": str(args.out / "results.json")}, indent=2))
    elif args.command == "prepare-agent-data":
        from .agent_data import combine_agent_runs
        manifest = combine_agent_runs(args.runs, args.out, seed=args.seed, test_fraction=args.test_fraction)
        print(json.dumps({"unique_inputs": manifest["unique_inputs"], "splits": {
            name: {"rows": split["rows"], "label_counts": split["label_counts"]}
            for name, split in manifest["splits"].items()}}, indent=2))
    elif args.command == "run-agent-comparison":
        from .compare import run_agent_comparison
        options = vars(args).copy()
        options.pop("command")
        result = run_agent_comparison(**options)
        print(json.dumps({"status": result["status"],
                          "head_to_head": result["head_to_head"],
                          "results": str(args.out / "results.json")}, indent=2))
    elif args.command == "verify":
        from .verify import verify_release
        print(json.dumps(verify_release(args.root), indent=2))
    else:
        from .data import LABELS, input_id, write_jsonl
        rows = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
        if not rows:
            parser.error("Prediction data is empty")
        metadata = json.loads((args.checkpoint / "monitor.json").read_text())
        if metadata.get("task") == "injection_detection":
            from .injection_mlp import load_injection_mlp, _predictions
            predictor = load_injection_mlp(args.checkpoint)
            output = _predictions(rows, predictor.predict_proba(rows))
        else:
            from .banking import load_banking_mlp
            cause, correct = load_banking_mlp(args.checkpoint).predict(rows)
            output = [{"input_id": input_id(row), "correct_probability": float(ok),
                       "cause_probabilities": dict(zip(LABELS, map(float, probs)))}
                      for row, probs, ok in zip(rows, cause, correct)]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.out, output)
        print(f"Wrote {len(output)} predictions to {args.out}")


if __name__ == "__main__":
    main()
