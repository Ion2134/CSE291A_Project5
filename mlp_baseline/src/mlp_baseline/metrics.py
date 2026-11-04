"""Probability helpers, temperature calibration, and the banking monitor's evaluation metrics."""
from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)

from .common import LABELS, cause_of

NONE_INDEX = LABELS.index("none")
CORRECT_THRESHOLD = 0.5          # P(correct) at or above this means "output is correct".
ECE_BINS = 10

# Temperature search: below this many rows (or a single class), probabilities stay uncalibrated.
MIN_CALIBRATION_ROWS = 20
TEMPERATURE_GRID = np.geomspace(0.25, 4, 101)


# ---------------------------------------------------------------- probabilities and calibration

def probabilities(values):
    """Validate and row-normalize a probability matrix."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all() or (values < 0).any() or (values.sum(axis=1) <= 0).any():
        raise ValueError("Invalid model probabilities")
    return values / values.sum(axis=1, keepdims=True)


def as_binary(correct_probs):
    """``P(correct)`` as a two-column ``[P(wrong), P(correct)]`` matrix."""
    return np.stack([1 - correct_probs, correct_probs], axis=1)


def scale_temperature(values, temperature=1.0):
    logits = np.log(np.clip(probabilities(values), 1e-9, 1)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    return probabilities(np.exp(logits))


def fit_temperature(values, targets):
    """The temperature on ``TEMPERATURE_GRID`` with the lowest log loss on ``targets``."""
    values = probabilities(values)
    targets = np.asarray(targets, dtype=int)
    if len(targets) < MIN_CALIBRATION_ROWS or len(set(targets)) < 2:
        return {"temperature": 1.0, "fitted": False, "samples": len(targets),
                "reason": "Need at least 20 validation examples and two classes; probabilities are uncalibrated."}

    losses = [-np.log(np.clip(scale_temperature(values, t)[np.arange(len(targets)), targets], 1e-9, 1)).mean()
              for t in TEMPERATURE_GRID]
    return {"temperature": float(TEMPERATURE_GRID[int(np.argmin(losses))]), "fitted": True, "samples": len(targets)}


def fit_calibration(cause_probs, rows, source: str) -> dict:
    """Fit the cause temperature and the correctness temperature (from raw P(cause=none))."""
    correct = cause_probs[:, 0]
    return {
        "cause": fit_temperature(cause_probs, [LABELS.index(cause_of(row)) for row in rows]),
        "correctness": fit_temperature(as_binary(correct), [int(row["expected"]["output_correct"]) for row in rows]),
        "source": source,
    }


def apply_calibration(cause_probs, calibration):
    """Scale cause probabilities, then correctness (from the calibrated P(cause=none)).

    Returns ``(cause probabilities, P(output correct))``.
    """
    cause = scale_temperature(cause_probs, calibration["cause"]["temperature"])
    binary = scale_temperature(as_binary(cause[:, 0]), calibration["correctness"]["temperature"])
    return cause, binary[:, 1]


def majority_probabilities(train_labels, n_rows: int, labels):
    """One-hot rows that always predict the most common training label (the majority baseline)."""
    majority = Counter(train_labels).most_common(1)[0][0]
    return np.tile(np.eye(len(labels))[labels.index(majority)], (n_rows, 1))


# ---------------------------------------------------------------- banking monitor metrics

def evaluate(rows, cause_probs, correct_probs, latencies_ms=None) -> dict:
    """All banking-monitor metrics: cause classification, error-type attribution, error detection."""
    cause_probs = probabilities(cause_probs)
    correct_probs = np.asarray(correct_probs, dtype=float)
    if (cause_probs.shape != (len(rows), len(LABELS)) or correct_probs.shape != (len(rows),)
            or not np.isfinite(correct_probs).all() or ((correct_probs < 0) | (correct_probs > 1)).any()):
        raise ValueError("Predictions must match the dataset and valid monitor output shapes")

    cause_true = np.array([LABELS.index(cause_of(row)) for row in rows])
    cause_pred = cause_probs.argmax(axis=1)
    correct_true = np.array([row["expected"]["output_correct"] for row in rows])
    correct_pred = correct_probs >= CORRECT_THRESHOLD

    result = {
        "samples": len(rows),
        "cause_labels": list(LABELS),
        **_cause_metrics(cause_true, cause_pred),
        **_error_type_metrics(cause_true, cause_pred),
        **_correctness_metrics(correct_true, correct_pred, correct_probs),
        "inconsistent_answers": int(((cause_pred == NONE_INDEX) != correct_pred).sum()),
    }
    if latencies_ms:
        result["latency_ms"] = {"median": float(np.median(latencies_ms)), "p95": float(np.percentile(latencies_ms, 95)),
                                "measurement": "one tool-call example per request, warm model, includes preprocessing"}
    return result


def _cause_metrics(cause_true, cause_pred) -> dict:
    labels = range(len(LABELS))
    precision, recall, f1, support = precision_recall_fscore_support(cause_true, cause_pred, labels=labels,
                                                                     zero_division=0)
    return {
        "cause_accuracy": float(accuracy_score(cause_true, cause_pred)),
        "cause_macro_f1": float(f1_score(cause_true, cause_pred, labels=labels, average="macro", zero_division=0)),
        "cause_per_class": {label: {"precision": float(precision[i]), "recall": float(recall[i]),
                                    "f1": float(f1[i]), "support": int(support[i])}
                            for i, label in enumerate(LABELS)},
        "cause_confusion_matrix": confusion_matrix(cause_true, cause_pred, labels=labels).tolist(),
    }


def _error_type_metrics(cause_true, cause_pred) -> dict:
    """Given a true error, was its exact cause predicted? Missed errors count as wrong in the first
    metric and are left out of the second."""
    errors = cause_true != NONE_INDEX
    detected = errors & (cause_pred != NONE_INDEX)
    return {
        "error_type_accuracy": float((cause_pred[errors] == cause_true[errors]).mean()) if errors.any() else None,
        "error_type_support": int(errors.sum()),
        "error_type_accuracy_when_detected": (float((cause_pred[detected] == cause_true[detected]).mean())
                                              if detected.any() else None),
        "error_type_detected_support": int(detected.sum()),
    }


def _correctness_metrics(correct_true, correct_pred, correct_probs) -> dict:
    wrong_true, wrong_pred = ~correct_true, ~correct_pred
    n_correct, n_wrong = int(correct_true.sum()), int(wrong_true.sum())
    return {
        "correctness_accuracy": float(accuracy_score(correct_true, correct_pred)),
        "error_detection_f1": float(f1_score(wrong_true, wrong_pred, zero_division=0)),
        "error_auroc": float(roc_auc_score(wrong_true, 1 - correct_probs)) if n_correct and n_wrong else None,
        "false_alarm_rate": float((correct_true & wrong_pred).sum() / n_correct) if n_correct else None,
        "missed_error_rate": float((wrong_true & correct_pred).sum() / n_wrong) if n_wrong else None,
        "correctness_brier": float(((correct_probs - correct_true) ** 2).mean()),
        "correctness_ece": _expected_calibration_error(correct_true, correct_pred, correct_probs),
    }


def _expected_calibration_error(correct_true, correct_pred, correct_probs) -> float:
    """Gap between confidence and accuracy of the correct/wrong answer, averaged over confidence bins."""
    confidence = as_binary(correct_probs).max(axis=1)
    right = correct_true == correct_pred
    edges = np.linspace(0, 1, ECE_BINS + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (confidence >= low) & (confidence < high if high < 1 else confidence <= high)
        if selected.any():
            ece += float(selected.mean() * abs(confidence[selected].mean() - right[selected].mean()))
    return ece
