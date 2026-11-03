"""Head-to-head first-party evaluation: Scenario A vs Scenario B.

Scenario A trains a three-class MLP on the combined first-party dataset only.
Scenario B stacks a first-party logistic regression on the frozen external
binary injection model. Both use the same group-disjoint folds for model
selection, fit temperatures on out-of-fold train-pool predictions, and are
evaluated once on the same held-out first-party test rows. External data
trains Scenario B's frozen base only; no external row is ever evaluated.
"""
from __future__ import annotations

import csv
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .agent_data import _check_row, validate_combined_dataset
from .banking_data import _read_jsonl
from .cv import group_kfold
from .data import LABELS, input_id, request_group, write_json, write_jsonl
from .injection_mlp import load_injection_mlp
from .metrics import evaluate, fit_temperature, scale_temperature
from .mlp import load_mlp, train_mlp
from .stacking import StackedMonitor, fit_stacker, save_stacker

CV_METRICS = ("cause_macro_f1", "error_auroc", "error_type_accuracy", "error_detection_f1",
              "false_alarm_rate", "missed_error_rate", "correctness_brier")


def _binary_correct(cause_probs):
    return cause_probs[:, 0]


def _fit_calibrations(oof_cause, train_rows):
    targets = [LABELS.index(row["expected"]["cause"]) for row in train_rows]
    correct = _binary_correct(oof_cause)
    calibration = {
        "cause": fit_temperature(oof_cause, targets),
        "correctness": fit_temperature(np.stack([1 - correct, correct], axis=1),
                                       [int(row["expected"]["output_correct"]) for row in train_rows]),
        "source": "out-of-fold train-pool predictions only",
    }
    return calibration


def _apply_calibrations(cause_probs, calibration):
    cause = scale_temperature(cause_probs, calibration["cause"]["temperature"])
    correct = _binary_correct(cause)
    binary = scale_temperature(np.stack([1 - correct, correct], axis=1),
                               calibration["correctness"]["temperature"])
    return cause, binary[:, 1]


def _per_source_run_metrics(test_rows, causes, corrects):
    metrics = {}
    runs = sorted({run for row in test_rows for run in row["meta"]["source_runs"]})
    for run in runs:
        selected = [index for index, row in enumerate(test_rows) if run in row["meta"]["source_runs"]]
        metrics[run] = evaluate([test_rows[index] for index in selected],
                                causes[selected], corrects[selected])
    return metrics


def _cv_summary(fold_metrics):
    summary = {}
    for key in CV_METRICS:
        values = [metric[key] for metric in fold_metrics if metric[key] is not None]
        if values:
            summary[key] = {"mean": float(np.mean(values)), "std": float(np.std(values)),
                            "folds": len(values), "per_fold": [float(value) for value in values]}
        else:
            summary[key] = {"mean": None, "std": None, "folds": 0, "per_fold": []}
    return summary


def _scenario_a(folds, train_rows, test_rows, models_dir: Path, *, seed, epochs, batch_size,
                max_features, verbose):
    fold_metrics, fold_epochs, oof = [], [], [None] * len(train_rows)
    train_index = {input_id(row): index for index, row in enumerate(train_rows)}
    for fold in folds:
        fold_dir = models_dir / "folds" / fold["name"]
        report = train_mlp(fold["train"], fold["val"], fold_dir, seed=seed, epochs=epochs,
                           batch_size=batch_size, max_features=max_features, verbose=verbose)
        fold_epochs.append(report["best_epoch"])
        cause, correct = load_mlp(fold_dir).predict(fold["val"])
        fold_metrics.append(evaluate(fold["val"], cause, correct))
        for row, probs in zip(fold["val"], cause):
            oof[train_index[input_id(row)]] = probs
    selected = max(1, int(round(float(np.median(fold_epochs)))))
    oof_cause = np.asarray(oof)
    calibration = _fit_calibrations(oof_cause, train_rows)
    final_dir = models_dir / "final"
    report = train_mlp(train_rows, None, final_dir, seed=seed, epochs=selected,
                       batch_size=batch_size, max_features=max_features, verbose=verbose)
    write_json(final_dir / "monitor.json", {"kind": "mlp", "task": "banking_monitor",
                                            "labels": list(LABELS), "calibration": calibration})
    raw_cause, _ = load_mlp(final_dir).predict(test_rows)
    cause, correct = _apply_calibrations(raw_cause, calibration)
    details = {
        "kind": "mlp", "data": "first-party train pool only",
        "epoch_selection": {"per_fold_best_epoch": fold_epochs, "selected_epochs": selected},
        "cv_metrics": _cv_summary(fold_metrics), "calibration": calibration, "training": report,
    }
    return cause, correct, details


def _scenario_b(folds, train_rows, test_rows, models_dir: Path, *, seed, max_features,
                frozen, base_checkpoint):
    fold_metrics, oof = [], [None] * len(train_rows)
    train_index = {input_id(row): index for index, row in enumerate(train_rows)}
    for fold in folds:
        vectorizer, stacker = fit_stacker(fold["train"], frozen, seed=seed, max_features=max_features)
        monitor = StackedMonitor(frozen, vectorizer, stacker,
                                 {"cause": {"temperature": 1.0}, "correctness": {"temperature": 1.0}})
        cause = monitor.cause_proba(fold["val"])
        fold_metrics.append(evaluate(fold["val"], cause, _binary_correct(cause)))
        for row, probs in zip(fold["val"], cause):
            oof[train_index[input_id(row)]] = probs
    oof_cause = np.asarray(oof)
    calibration = _fit_calibrations(oof_cause, train_rows)
    vectorizer, stacker = fit_stacker(train_rows, frozen, seed=seed, max_features=max_features)
    monitor = StackedMonitor(frozen, vectorizer, stacker, calibration)
    final_dir = models_dir / "final"
    save_stacker(final_dir, vectorizer, stacker, calibration, base_checkpoint=str(base_checkpoint),
                 training={"model": "multinomial logistic regression", "class_weight": "balanced",
                           "seed": seed, "max_features": max_features,
                           "features": "calibrated injection probability + TF-IDF",
                           "selection": "fixed hyperparameters; CV used for calibration and fold metrics"})
    cause, correct = monitor.predict(test_rows)
    details = {
        "kind": "stacked_logreg",
        "data": "external data trains the frozen binary base only; the stacker trains on the first-party train pool",
        "base_checkpoint": str(base_checkpoint),
        "cv_metrics": _cv_summary(fold_metrics), "calibration": calibration,
    }
    return cause, correct, details


def run_agent_comparison(data: Path, out: Path, *, seed=42, k=5, epochs=40, batch_size=16,
                         max_features=4096, threads=4,
                         injection_checkpoint=Path("models/injection/seed42"), verbose=False):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new or empty comparison directory")
    torch.set_num_threads(threads)
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    splits, manifest = validate_combined_dataset(Path(data))
    train_rows, test_rows = splits["train"], splits["test"]
    folds_result = group_kfold(train_rows, k=k, seed=seed)
    folds = folds_result["folds"]
    cv_info = {"requested_k": folds_result["requested_k"], "k": folds_result["k"],
               "degraded": folds_result["degraded"],
               "degradation_reason": folds_result["degradation_reason"],
               "folds": [{"name": fold["name"], "train_rows": len(fold["train"]),
                          "val_rows": len(fold["val"]),
                          "val_label_counts": dict(sorted(Counter(
                              row["expected"]["cause"] for row in fold["val"]).items()))}
                         for fold in folds]}

    frozen = load_injection_mlp(Path(injection_checkpoint))
    cause_a, correct_a, details_a = _scenario_a(
        folds, train_rows, test_rows, out / "models" / "scenario-a-first-party-mlp",
        seed=seed, epochs=epochs, batch_size=batch_size, max_features=max_features, verbose=verbose)
    cause_b, correct_b, details_b = _scenario_b(
        folds, train_rows, test_rows, out / "models" / "scenario-b-hybrid-stacking",
        seed=seed, max_features=max_features, frozen=frozen, base_checkpoint=injection_checkpoint)

    scenarios = {"scenario_a_first_party_mlp": (cause_a, correct_a, details_a),
                 "scenario_b_hybrid_stacking": (cause_b, correct_b, details_b)}
    result_scenarios, prediction_rows = {}, []
    for name, (cause, correct, details) in scenarios.items():
        result_scenarios[name] = {**details, "test": evaluate(test_rows, cause, correct),
                                  "test_by_source_run": _per_source_run_metrics(test_rows, cause, correct)}
    for index, row in enumerate(test_rows):
        prediction_rows.append({
            "input_id": input_id(row), "source_runs": row["meta"]["source_runs"], "expected": row["expected"],
            "scenario_a": {"cause_probabilities": dict(zip(LABELS, map(float, cause_a[index]))),
                           "correct_probability": float(correct_a[index])},
            "scenario_b": {"cause_probabilities": dict(zip(LABELS, map(float, cause_b[index]))),
                           "correct_probability": float(correct_b[index])},
        })
    majority = Counter(row["expected"]["cause"] for row in train_rows).most_common(1)[0][0]
    majority_probs = np.tile(np.eye(3)[LABELS.index(majority)], (len(test_rows), 1))
    majority_baseline = evaluate(test_rows, majority_probs, majority_probs[:, 0])
    head_to_head = {key: {"scenario_a": result_scenarios["scenario_a_first_party_mlp"]["test"][key],
                          "scenario_b": result_scenarios["scenario_b_hybrid_stacking"]["test"][key],
                          "majority_baseline": majority_baseline[key]}
                    for key in ("cause_accuracy", "cause_macro_f1", "error_type_accuracy",
                                "error_type_accuracy_when_detected", "correctness_accuracy",
                                "error_detection_f1", "error_auroc", "false_alarm_rate",
                                "missed_error_rate", "correctness_brier", "correctness_ece")}
    result = {
        "status": "complete", "task": "banking_monitor", "seed": seed,
        "eval_policy": ("First-party agent data only. External datasets train the frozen binary base of "
                        "scenario B and are otherwise unused; no external row is evaluated."),
        "device": "cpu", "data": manifest, "cv": cv_info,
        "scenarios": result_scenarios, "majority_baseline": majority_baseline,
        "head_to_head": head_to_head,
        "limitation": (f"{manifest['unique_inputs']} unique first-party examples and {len(test_rows)} held-out "
                       f"test rows with {sum(row['expected']['cause'] == 'agent' for row in test_rows)} agent-error "
                       "test example(s); treat as a development comparison, not a benchmark claim."),
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(out / "results.json", result)
    write_jsonl(out / "test_predictions.jsonl", prediction_rows)
    columns = ["model", "samples", "cause_accuracy", "cause_macro_f1", "error_type_accuracy",
               "error_type_accuracy_when_detected", "correctness_accuracy", "error_detection_f1",
               "error_auroc", "false_alarm_rate", "missed_error_rate", "correctness_brier", "correctness_ece"]
    with (out / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for name, metrics in [("majority_baseline", majority_baseline),
                              ("scenario_a_first_party_mlp", result_scenarios["scenario_a_first_party_mlp"]["test"]),
                              ("scenario_b_hybrid_stacking", result_scenarios["scenario_b_hybrid_stacking"]["test"])]:
            writer.writerow({"model": name, **{key: metrics[key] for key in columns[1:]}})
    return result


def fit_predict(train: Path, test: Path, out: Path, *, model="mlp", validation: Path | None = None,
                seed=42, k=5, epochs=40, batch_size=16, max_features=4096, threads=4,
                injection_checkpoint=Path("models/injection/seed42"), keep_models=False, verbose=False):
    """Train one scenario on a given train file and predict a given test file.

    This is the entry point the unified tester (``testing/sweep.py``) calls for
    each fold. Training is the same as ``run_agent_comparison``: inner group
    cross-validation picks the MLP epoch count and fits calibration on
    out-of-fold predictions, then the model is refit on every training row.
    ``validation`` rows (if given) are added to the training rows, because the
    inner cross-validation already holds data out.
    """
    if model not in ("mlp", "stacking"):
        raise ValueError("model must be 'mlp' or 'stacking'")
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new or empty output directory")
    torch.set_num_threads(threads)
    started = time.perf_counter()

    train_rows = _read_jsonl(Path(train))
    if validation is not None:
        train_rows += _read_jsonl(Path(validation))
    test_rows = _read_jsonl(Path(test))
    if not train_rows or not test_rows:
        raise ValueError("Train and test files must not be empty")
    questions = train_rows[0]["questions"]
    for name, rows in (("train", train_rows), ("test", test_rows)):
        for line, row in enumerate(rows, 1):
            _check_row(row, name, line, questions)
    if {request_group(row) for row in train_rows} & {request_group(row) for row in test_rows}:
        raise ValueError("A user-request group appears in both train and test")

    out.mkdir(parents=True, exist_ok=True)
    folds_result = group_kfold(train_rows, k=k, seed=seed)
    models_dir = out / "models"
    if model == "mlp":
        cause, correct, details = _scenario_a(
            folds_result["folds"], train_rows, test_rows, models_dir, seed=seed, epochs=epochs,
            batch_size=batch_size, max_features=max_features, verbose=verbose)
    else:
        frozen = load_injection_mlp(Path(injection_checkpoint))
        cause, correct, details = _scenario_b(
            folds_result["folds"], train_rows, test_rows, models_dir, seed=seed,
            max_features=max_features, frozen=frozen, base_checkpoint=injection_checkpoint)
    if not keep_models:
        shutil.rmtree(models_dir)

    write_jsonl(out / "test_predictions.jsonl", [
        {"input_id": input_id(row), "expected": row["expected"],
         "cause_probabilities": dict(zip(LABELS, map(float, probs))),
         "correct_probability": float(ok)}
        for row, probs, ok in zip(test_rows, cause, correct)])
    details.pop("training", None)  # Per-epoch history is large and not needed per fold.
    result = {
        "model": model, "seed": seed, "train_rows": len(train_rows), "test_rows": len(test_rows),
        "inner_cv": {"requested_k": folds_result["requested_k"], "k": folds_result["k"],
                     "degraded": folds_result["degraded"]},
        "details": details, "test": evaluate(test_rows, cause, correct),
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(out / "results.json", result)
    return result
