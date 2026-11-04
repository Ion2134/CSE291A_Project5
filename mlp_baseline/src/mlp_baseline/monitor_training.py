"""Train the two banking monitors on one train/test split.

- ``mlp`` (scenario A): a three-class MLP trained on first-party rows only.
- ``stacking`` (scenario B): a logistic regression on the frozen injection detector's
  P(injection) plus TF-IDF features.

Both run inner group cross-validation on the training rows: the MLP uses it to pick its epoch
count, and both fit their temperatures on the out-of-fold predictions. The final model is then
refit on every training row. ``fit_predict`` is what the unified tester calls for each fold.
"""
from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch

from .common import (DEFAULT_INJECTION_CHECKPOINT, LABELS, check_monitor_row, input_id, read_jsonl, request_group,
                     require_empty_dir, write_json, write_jsonl)
from .injection_model import InjectionMLPPredictor, load_injection_mlp
from .metrics import apply_calibration, evaluate, fit_calibration
from .mlp import MONITOR_FILE, TrainSettings, load_mlp, train_mlp
from .splits import group_kfold
from .stacking import StackedMonitor, fit_stacker, save_stacker

MODELS = ("mlp", "stacking")
OUT_OF_FOLD = "out-of-fold train-pool predictions only"
# Fold metrics summarized in each scenario's ``cv_metrics``.
CV_METRICS = ("cause_macro_f1", "error_auroc", "error_type_accuracy", "error_detection_f1",
              "false_alarm_rate", "missed_error_rate", "correctness_brier")
STACKER_TRAINING = {"model": "multinomial logistic regression", "class_weight": "balanced"}


@dataclass(frozen=True)
class MonitorSettings:
    """How to train either monitor."""

    train: TrainSettings = field(default_factory=TrainSettings)
    k: int = 5                                                # Inner cross-validation folds.
    injection_checkpoint: Path = DEFAULT_INJECTION_CHECKPOINT  # The stacking monitor's frozen base.


@dataclass
class ScenarioOutput:
    """One trained scenario's calibrated test predictions and how it was trained."""

    cause: np.ndarray      # (rows, 3) cause probabilities in LABELS order
    correct: np.ndarray    # (rows,) P(output correct)
    details: dict          # training, calibration and CV summary, written into results.json


# ---------------------------------------------------------------- fit-predict (one fold of the tester)

def fit_predict(train: Path, test: Path, out: Path, settings: MonitorSettings = MonitorSettings(), *,
                model="mlp", validation: Path | None = None, keep_models=False) -> dict:
    """Train one monitor on ``train`` (+ ``validation``) and write predictions for ``test``.

    ``validation`` rows are added to training, because the inner cross-validation already holds
    data out. Fold models are deleted afterwards unless ``keep_models``.
    """
    if model not in MODELS:
        raise ValueError("model must be 'mlp' or 'stacking'")
    out = Path(out)
    require_empty_dir(out, "Use a new or empty output directory")
    torch.set_num_threads(settings.train.threads)
    started = time.perf_counter()

    train_rows = read_jsonl(Path(train)) + (read_jsonl(Path(validation)) if validation is not None else [])
    test_rows = read_jsonl(Path(test))
    _check_fold(train_rows, test_rows)

    out.mkdir(parents=True, exist_ok=True)
    folds_result = group_kfold(train_rows, k=settings.k, seed=settings.train.seed)
    models_dir = out / "models"
    if model == "mlp":
        output = train_first_party_mlp(folds_result["folds"], train_rows, test_rows, models_dir, settings.train)
    else:
        frozen = load_injection_mlp(Path(settings.injection_checkpoint))
        output = train_stacking(folds_result["folds"], train_rows, test_rows, models_dir, settings, frozen)
    if not keep_models:
        shutil.rmtree(models_dir)

    write_jsonl(out / "test_predictions.jsonl", [
        {"input_id": input_id(row), "expected": row["expected"],
         "cause_probabilities": dict(zip(LABELS, map(float, probs))), "correct_probability": float(ok)}
        for row, probs, ok in zip(test_rows, output.cause, output.correct)])

    output.details.pop("training", None)  # Per-epoch history is large and not needed per fold.
    result = {
        "model": model, "seed": settings.train.seed, "train_rows": len(train_rows), "test_rows": len(test_rows),
        "inner_cv": {key: folds_result[key] for key in ("requested_k", "k", "degraded")},
        "details": output.details, "test": evaluate(test_rows, output.cause, output.correct),
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(out / "results.json", result)
    return result


def _check_fold(train_rows, test_rows) -> None:
    """Both parts are non-empty, share one schema, have valid labels, and share no request group."""
    if not train_rows or not test_rows:
        raise ValueError("Train and test files must not be empty")
    questions = train_rows[0]["questions"]
    for name, rows in (("train", train_rows), ("test", test_rows)):
        for line, row in enumerate(rows, 1):
            check_monitor_row(row, name, line, questions)
    if {request_group(row) for row in train_rows} & {request_group(row) for row in test_rows}:
        raise ValueError("A user-request group appears in both train and test")


# ---------------------------------------------------------------- the two scenarios

def train_first_party_mlp(folds, train_rows, test_rows, models_dir: Path, settings: TrainSettings) -> ScenarioOutput:
    """Scenario A: cross-validate to pick the epoch count and calibration, then refit on all rows."""
    fold_epochs = []

    def fit_fold(fold):
        fold_dir = models_dir / "folds" / fold["name"]
        report = train_mlp(fold["train"], fold["val"], fold_dir, settings)
        fold_epochs.append(report["best_epoch"])
        cause, _ = load_mlp(fold_dir).predict(fold["val"])
        return cause

    fold_metrics, oof_cause = _cross_validate(folds, train_rows, fit_fold)
    selected = max(1, int(round(float(np.median(fold_epochs)))))
    calibration = fit_calibration(oof_cause, train_rows, OUT_OF_FOLD)

    final_dir = models_dir / "final"
    report = train_mlp(train_rows, None, final_dir, replace(settings, epochs=selected))
    write_json(final_dir / MONITOR_FILE, {"kind": "mlp", "task": "banking_monitor", "labels": list(LABELS),
                                          "calibration": calibration})
    raw_cause, _ = load_mlp(final_dir).predict(test_rows)
    cause, correct = apply_calibration(raw_cause, calibration)

    details = {
        "kind": "mlp", "data": "first-party train pool only",
        "epoch_selection": {"per_fold_best_epoch": fold_epochs, "selected_epochs": selected},
        "cv_metrics": _cv_summary(fold_metrics), "calibration": calibration, "training": report,
    }
    return ScenarioOutput(cause, correct, details)


def train_stacking(folds, train_rows, test_rows, models_dir: Path, settings: MonitorSettings,
                   frozen: InjectionMLPPredictor) -> ScenarioOutput:
    """Scenario B: fixed hyperparameters; cross-validation only fits the calibration."""
    seed, max_features = settings.train.seed, settings.train.max_features
    uncalibrated = {"cause": {"temperature": 1.0}, "correctness": {"temperature": 1.0}}

    def fit_fold(fold):
        vectorizer, stacker = fit_stacker(fold["train"], frozen, seed=seed, max_features=max_features)
        return StackedMonitor(frozen, vectorizer, stacker, uncalibrated).cause_proba(fold["val"])

    fold_metrics, oof_cause = _cross_validate(folds, train_rows, fit_fold)
    calibration = fit_calibration(oof_cause, train_rows, OUT_OF_FOLD)

    vectorizer, stacker = fit_stacker(train_rows, frozen, seed=seed, max_features=max_features)
    monitor = StackedMonitor(frozen, vectorizer, stacker, calibration)
    base_checkpoint = str(settings.injection_checkpoint)
    save_stacker(models_dir / "final", vectorizer, stacker, calibration, base_checkpoint=base_checkpoint,
                 training={**STACKER_TRAINING, "seed": seed, "max_features": max_features,
                           "features": "calibrated injection probability + TF-IDF",
                           "selection": "fixed hyperparameters; CV used for calibration and fold metrics"})
    cause, correct = monitor.predict(test_rows)

    details = {
        "kind": "stacked_logreg",
        "data": "external data trains the frozen binary base only; the stacker trains on the first-party train pool",
        "base_checkpoint": base_checkpoint,
        "cv_metrics": _cv_summary(fold_metrics), "calibration": calibration,
    }
    return ScenarioOutput(cause, correct, details)


def _cross_validate(folds, train_rows, fit_fold):
    """Run ``fit_fold(fold) -> validation cause probabilities`` on each fold.

    Returns each fold's metrics and the out-of-fold cause probabilities in ``train_rows`` order.
    """
    position = {input_id(row): index for index, row in enumerate(train_rows)}
    fold_metrics, oof = [], [None] * len(train_rows)
    for fold in folds:
        cause = fit_fold(fold)
        fold_metrics.append(evaluate(fold["val"], cause, cause[:, 0]))
        for row, probs in zip(fold["val"], cause):
            oof[position[input_id(row)]] = probs
    return fold_metrics, np.asarray(oof)


def _cv_summary(fold_metrics) -> dict:
    """Mean, std and per-fold values of each ``CV_METRICS`` entry (skipping undefined folds)."""
    summary = {}
    for key in CV_METRICS:
        values = [metric[key] for metric in fold_metrics if metric[key] is not None]
        if values:
            summary[key] = {"mean": float(np.mean(values)), "std": float(np.std(values)),
                            "folds": len(values), "per_fold": [float(value) for value in values]}
        else:
            summary[key] = {"mean": None, "std": None, "folds": 0, "per_fold": []}
    return summary
