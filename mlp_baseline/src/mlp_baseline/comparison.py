"""``mlp-baseline run-agent-comparison``: both monitors on one combined dataset's fixed split.

Scenario A (``mlp``) and scenario B (``stacking``) share the same inner cross-validation folds,
are trained on the train pool and scored once on the same held-out test rows, next to a
majority baseline. Superseded by the unified tester (many splits); kept to reproduce
``archive/agent-comparison``.
"""
from __future__ import annotations

import csv
import time
from collections import Counter
from pathlib import Path

import torch

from .common import LABELS, cause_of, input_id, require_empty_dir, write_json, write_jsonl
from .injection_model import load_injection_mlp
from .metrics import evaluate, majority_probabilities
from .monitor_datasets import validate_combined_dataset
from .monitor_training import MonitorSettings, ScenarioOutput, train_first_party_mlp, train_stacking
from .splits import group_kfold

SCENARIO_A, SCENARIO_B = "scenario_a_first_party_mlp", "scenario_b_hybrid_stacking"
SCENARIO_DIRS = {SCENARIO_A: "scenario-a-first-party-mlp", SCENARIO_B: "scenario-b-hybrid-stacking"}
# Test metrics in the head-to-head table and metrics.csv, in column order.
REPORTED_METRICS = ("cause_accuracy", "cause_macro_f1", "error_type_accuracy",
                    "error_type_accuracy_when_detected", "correctness_accuracy", "error_detection_f1",
                    "error_auroc", "false_alarm_rate", "missed_error_rate", "correctness_brier",
                    "correctness_ece")
EVAL_POLICY = ("First-party agent data only. External datasets train the frozen binary base of "
               "scenario B and are otherwise unused; no external row is evaluated.")


def run_agent_comparison(data: Path, out: Path, settings: MonitorSettings = MonitorSettings()) -> dict:
    """Train both scenarios on a combined dataset's train pool and score them on its test split."""
    out = Path(out)
    require_empty_dir(out, "Use a new or empty comparison directory")
    torch.set_num_threads(settings.train.threads)
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)

    splits, manifest = validate_combined_dataset(Path(data))
    train_rows, test_rows = splits["train"], splits["test"]
    folds_result = group_kfold(train_rows, k=settings.k, seed=settings.train.seed)
    folds = folds_result["folds"]
    models_dir = out / "models"

    frozen = load_injection_mlp(Path(settings.injection_checkpoint))
    outputs = {
        SCENARIO_A: train_first_party_mlp(folds, train_rows, test_rows, models_dir / SCENARIO_DIRS[SCENARIO_A],
                                          settings.train),
        SCENARIO_B: train_stacking(folds, train_rows, test_rows, models_dir / SCENARIO_DIRS[SCENARIO_B],
                                   settings, frozen),
    }

    scenarios = {name: {**output.details, "test": evaluate(test_rows, output.cause, output.correct),
                        "test_by_source_run": _per_source_run_metrics(test_rows, output)}
                 for name, output in outputs.items()}
    majority_probs = majority_probabilities([cause_of(row) for row in train_rows], len(test_rows), LABELS)
    majority_baseline = evaluate(test_rows, majority_probs, majority_probs[:, 0])

    result = {
        "status": "complete", "task": "banking_monitor", "seed": settings.train.seed,
        "eval_policy": EVAL_POLICY,
        "device": "cpu", "data": manifest, "cv": _cv_info(folds_result),
        "scenarios": scenarios, "majority_baseline": majority_baseline,
        "head_to_head": _head_to_head(scenarios, majority_baseline),
        "limitation": _limitation(manifest, test_rows),
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(out / "results.json", result)
    write_jsonl(out / "test_predictions.jsonl", _comparison_predictions(test_rows, outputs))
    _write_metrics_csv(out / "metrics.csv", {"majority_baseline": majority_baseline,
                                             SCENARIO_A: scenarios[SCENARIO_A]["test"],
                                             SCENARIO_B: scenarios[SCENARIO_B]["test"]})
    return result


def _per_source_run_metrics(test_rows, output: ScenarioOutput) -> dict:
    """The test metrics restricted to rows from each agent run."""
    metrics = {}
    for run in sorted({run for row in test_rows for run in row["meta"]["source_runs"]}):
        selected = [index for index, row in enumerate(test_rows) if run in row["meta"]["source_runs"]]
        metrics[run] = evaluate([test_rows[index] for index in selected],
                                output.cause[selected], output.correct[selected])
    return metrics


def _head_to_head(scenarios: dict, majority_baseline: dict) -> dict:
    return {key: {"scenario_a": scenarios[SCENARIO_A]["test"][key],
                  "scenario_b": scenarios[SCENARIO_B]["test"][key],
                  "majority_baseline": majority_baseline[key]}
            for key in REPORTED_METRICS}


def _limitation(manifest: dict, test_rows) -> str:
    agent_errors = sum(cause_of(row) == "agent" for row in test_rows)
    return (f"{manifest['unique_inputs']} unique first-party examples and {len(test_rows)} held-out "
            f"test rows with {agent_errors} agent-error "
            "test example(s); treat as a development comparison, not a benchmark claim.")


def _cv_info(folds_result: dict) -> dict:
    return {"requested_k": folds_result["requested_k"], "k": folds_result["k"],
            "degraded": folds_result["degraded"],
            "degradation_reason": folds_result["degradation_reason"],
            "folds": [{"name": fold["name"], "train_rows": len(fold["train"]), "val_rows": len(fold["val"]),
                       "val_label_counts": dict(sorted(Counter(cause_of(row) for row in fold["val"]).items()))}
                      for fold in folds_result["folds"]]}


def _comparison_predictions(test_rows, outputs: dict[str, ScenarioOutput]) -> list[dict]:
    """One row per test example with both scenarios' probabilities."""
    def probs(name, index):
        output = outputs[name]
        return {"cause_probabilities": dict(zip(LABELS, map(float, output.cause[index]))),
                "correct_probability": float(output.correct[index])}

    return [{"input_id": input_id(row), "source_runs": row["meta"]["source_runs"], "expected": row["expected"],
             "scenario_a": probs(SCENARIO_A, index), "scenario_b": probs(SCENARIO_B, index)}
            for index, row in enumerate(test_rows)]


def _write_metrics_csv(path: Path, by_model: dict) -> None:
    columns = ["model", "samples", *REPORTED_METRICS]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for name, metrics in by_model.items():
            writer.writerow({"model": name, **{key: metrics[key] for key in columns[1:]}})
