"""The ``mlp-baseline`` command line: one subcommand per pipeline, each with a small handler.

Handlers import their pipeline lazily, so ``--help`` and light commands don't load torch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import DEFAULT_INJECTION_CHECKPOINT


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.handler(args, parser)


# ---------------------------------------------------------------- handlers

def _prepare_injection(args, parser):
    from .injection_data import prepare_injection_data
    report = prepare_injection_data(args.cache, args.out, seed=args.seed, threshold=args.similarity,
                                    download=args.download)
    print(json.dumps(report["splits"], indent=2))


def _train_injection(args, parser):
    from .injection_training import train_injection_mlp
    train_injection_mlp(args.data, args.out, _train_settings(args))


def _prepare_banking(args, parser):
    from .banking_dev import prepare_training_data
    print(json.dumps(prepare_training_data(args.run, args.out), indent=2))


def _train_banking(args, parser):
    from .banking_dev import run_banking
    result = run_banking(args.run, args.out, _train_settings(args))
    print(json.dumps({"status": result["status"], "results": str(args.out / "results.json")}, indent=2))


def _prepare_agent_data(args, parser):
    from .agent_runs import COMBINED_RUNS, read_runs_config
    from .monitor_datasets import combine_agent_runs
    runs = read_runs_config(args.config)["runs"] if args.config else COMBINED_RUNS
    manifest = combine_agent_runs(args.runs, args.out, seed=args.seed, test_fraction=args.test_fraction, runs=runs)
    print(json.dumps({"unique_inputs": manifest["unique_inputs"], "splits": _split_sizes(manifest)}, indent=2))


def _export_laya_banking(args, parser):
    from .monitor_datasets import export_laya_banking
    manifest = export_laya_banking(args.data, args.out, seed=args.seed, val_fraction=args.val_fraction)
    print(json.dumps({"source_combined": manifest["source_combined"], "splits": _split_sizes(manifest)}, indent=2))


def _run_agent_comparison(args, parser):
    from .comparison import run_agent_comparison
    result = run_agent_comparison(args.data, args.out, _monitor_settings(args))
    print(json.dumps({"status": result["status"], "head_to_head": result["head_to_head"],
                      "results": str(args.out / "results.json")}, indent=2))


def _fit_predict(args, parser):
    from .monitor_training import fit_predict
    result = fit_predict(args.train, args.test, args.out, _monitor_settings(args), model=args.model,
                         validation=args.validation, keep_models=args.keep_models)
    print(json.dumps({"model": result["model"], "test_rows": result["test_rows"],
                      "predictions": str(args.out / "test_predictions.jsonl")}, indent=2))


def _predict(args, parser):
    from .common import LABELS, input_id, read_json, read_jsonl, write_jsonl
    from .mlp import MONITOR_FILE
    rows = read_jsonl(args.data)
    if not rows:
        parser.error("Prediction data is empty")

    if read_json(args.checkpoint / MONITOR_FILE).get("task") == "injection_detection":
        from .injection_model import load_injection_mlp, prediction_rows
        output = prediction_rows(rows, load_injection_mlp(args.checkpoint).predict_proba(rows))
    else:
        from .banking_dev import load_banking_mlp
        cause, correct = load_banking_mlp(args.checkpoint).predict(rows)
        output = [{"input_id": input_id(row), "correct_probability": float(ok),
                   "cause_probabilities": dict(zip(LABELS, map(float, probs)))}
                  for row, probs, ok in zip(rows, cause, correct)]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, output)
    print(f"Wrote {len(output)} predictions to {args.out}")


def _verify(args, parser):
    from .verify import verify_release
    print(json.dumps(verify_release(args.root), indent=2))


# ---------------------------------------------------------------- helpers

def _train_settings(args):
    from .mlp import TrainSettings
    # Not every command has these flags; the TrainSettings defaults apply when it doesn't.
    optional = {name: getattr(args, name)
                for name in ("max_features", "patience", "verbose", "use_checks", "embedding") if hasattr(args, name)}
    return TrainSettings(seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, threads=args.threads,
                         **optional)


def _monitor_settings(args):
    from .monitor_training import MonitorSettings
    return MonitorSettings(_train_settings(args), k=args.k, injection_checkpoint=args.injection_checkpoint,
                           view=getattr(args, "view", "full"))


def _split_sizes(manifest: dict) -> dict:
    return {name: {"rows": split["rows"], "label_counts": split["label_counts"]}
            for name, split in manifest["splits"].items()}


# ---------------------------------------------------------------- arguments

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone local MLP baseline")
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name, handler, description):
        sub = commands.add_parser(name, help=description)
        sub.set_defaults(handler=handler)
        return sub

    sub = command("prepare-injection", _prepare_injection, "Regenerate public-data splits from pinned sources")
    sub.add_argument("--cache", type=Path, default=Path(".cache/dataset-audit"))
    sub.add_argument("--out", type=Path, required=True)
    sub.add_argument("--seed", type=int, default=42)
    sub.add_argument("--similarity", type=float, default=0.9)
    sub.add_argument("--download", action="store_true")

    sub = command("train-injection", _train_injection, "Train the CPU injection-detection MLP")
    sub.add_argument("--data", type=Path, required=True)
    sub.add_argument("--out", type=Path, required=True)
    _add_training_arguments(sub, batch_size=64)
    sub.add_argument("--max-features", type=int, default=4096)
    sub.add_argument("--patience", type=int, default=8)

    sub = command("prepare-banking", _prepare_banking, "Repair known legacy labels and deduplicate a simulator run")
    sub.add_argument("--run", type=Path, required=True)
    sub.add_argument("--out", type=Path, required=True)

    sub = command("train-banking", _train_banking, "Train the separate banking development baseline")
    sub.add_argument("--run", type=Path, required=True)
    sub.add_argument("--out", type=Path, required=True)
    _add_training_arguments(sub, batch_size=16)

    sub = command("prepare-agent-data", _prepare_agent_data, "Combine first-party agent runs into one pruned dataset")
    sub.add_argument("--runs", type=Path, required=True)
    sub.add_argument("--out", type=Path, required=True)
    sub.add_argument("--seed", type=int, default=42)
    sub.add_argument("--test-fraction", type=float, default=0.2)
    sub.add_argument("--config", type=Path,
                     help="JSON config whose 'runs' list replaces COMBINED_RUNS (e.g. testing/configs/agent-v4.json)")

    sub = command("export-laya-banking", _export_laya_banking,
                  "Export a combined agent dataset as Laya train/val/test (shared test split)")
    sub.add_argument("--data", type=Path, required=True, help="agent-combined directory from prepare-agent-data")
    sub.add_argument("--out", type=Path, required=True)
    sub.add_argument("--seed", type=int, default=42)
    sub.add_argument("--val-fraction", type=float, default=0.2)

    sub = command("run-agent-comparison", _run_agent_comparison, "Head-to-head first-party MLP vs hybrid stacking")
    sub.add_argument("--data", type=Path, required=True)
    sub.add_argument("--out", type=Path, required=True)
    _add_monitor_arguments(sub)

    sub = command("fit-predict", _fit_predict, "Train one scenario on a train file and predict a test file")
    sub.add_argument("--model", choices=["mlp", "mlp_embed", "stacking", "rules", "rules_plus"],
                     required=True)
    sub.add_argument("--train", type=Path, required=True)
    sub.add_argument("--validation", type=Path, help="Optional; these rows are added to training")
    sub.add_argument("--test", type=Path, required=True)
    sub.add_argument("--out", type=Path, required=True)
    _add_monitor_arguments(sub)
    sub.add_argument("--keep-models", action="store_true")
    sub.add_argument("--embedding", help="Hugging Face model id; embed states with it instead of TF-IDF")
    sub.add_argument("--view", choices=["full", "middleware"], default="full",
                     help="middleware hides ledger_snapshot and received_args, which a real middleware cannot see")

    sub = command("predict", _predict, "Load an exported checkpoint and predict JSONL examples")
    sub.add_argument("--checkpoint", type=Path, required=True)
    sub.add_argument("--data", type=Path, required=True)
    sub.add_argument("--out", type=Path, required=True)

    sub = command("verify", _verify, "Check release hashes, dataset splits, and deployed predictions")
    sub.add_argument("--root", type=Path, default=Path("."))
    return parser


def _add_training_arguments(sub, *, batch_size: int) -> None:
    sub.add_argument("--seed", type=int, default=42)
    sub.add_argument("--epochs", type=int, default=40)
    sub.add_argument("--batch-size", type=int, default=batch_size)
    sub.add_argument("--threads", type=int, default=4)


def _add_monitor_arguments(sub) -> None:
    """Flags shared by ``run-agent-comparison`` and ``fit-predict``."""
    _add_training_arguments(sub, batch_size=16)
    sub.add_argument("--k", type=int, default=5, help="Inner cross-validation folds")
    sub.add_argument("--max-features", type=int, default=4096)
    sub.add_argument("--injection-checkpoint", type=Path, default=DEFAULT_INJECTION_CHECKPOINT)
    sub.add_argument("--verbose", action="store_true")
    sub.add_argument("--use-checks", action="store_true",
                     help="Add explicit comparison features (call vs mandate, args vs call, output vs ledger)")


if __name__ == "__main__":
    main()
