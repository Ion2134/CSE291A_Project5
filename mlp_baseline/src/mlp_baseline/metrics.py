from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)

from .data import LABELS


def probabilities(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all() or (values < 0).any() or (values.sum(axis=1) <= 0).any():
        raise ValueError("Invalid model probabilities")
    return values / values.sum(axis=1, keepdims=True)


def scale_temperature(values, temperature=1.0):
    logits = np.log(np.clip(probabilities(values), 1e-9, 1)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    return probabilities(np.exp(logits))


def fit_temperature(values, targets):
    values = probabilities(values)
    targets = np.asarray(targets, dtype=int)
    if len(targets) < 20 or len(set(targets)) < 2:
        return {"temperature": 1.0, "fitted": False, "samples": len(targets),
                "reason": "Need at least 20 validation examples and two classes; probabilities are uncalibrated."}
    candidates = np.geomspace(0.25, 4, 101)
    losses = [-np.log(np.clip(scale_temperature(values, t)[np.arange(len(targets)), targets], 1e-9, 1)).mean()
              for t in candidates]
    return {"temperature": float(candidates[int(np.argmin(losses))]), "fitted": True, "samples": len(targets)}


def evaluate(rows, cause_probs, correct_probs, latencies_ms=None):
    cause_probs = probabilities(cause_probs)
    correct_probs = np.asarray(correct_probs, dtype=float)
    if (cause_probs.shape != (len(rows), len(LABELS)) or correct_probs.shape != (len(rows),)
            or not np.isfinite(correct_probs).all() or ((correct_probs < 0) | (correct_probs > 1)).any()):
        raise ValueError("Predictions must match the dataset and valid monitor output shapes")
    cause_true = np.array([LABELS.index(row["expected"]["cause"]) for row in rows])
    correct_true = np.array([row["expected"]["output_correct"] for row in rows])
    cause_pred = cause_probs.argmax(axis=1)
    correct_pred = correct_probs >= 0.5
    precision, recall, f1, support = precision_recall_fscore_support(
        cause_true, cause_pred, labels=range(len(LABELS)), zero_division=0)
    wrong_true, wrong_pred = ~correct_true, ~correct_pred
    correct_count, wrong_count = int(correct_true.sum()), int(wrong_true.sum())
    none_index = LABELS.index("none")
    error_mask = cause_true != none_index
    error_support = int(error_mask.sum())
    detected_mask = error_mask & (cause_pred != none_index)
    detected_support = int(detected_mask.sum())
    binary = np.stack([1 - correct_probs, correct_probs], axis=1)
    confidence = binary.max(axis=1)
    right = correct_true == correct_pred
    ece = 0.0
    for low, high in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        selected = (confidence >= low) & (confidence < high if high < 1 else confidence <= high)
        if selected.any():
            ece += float(selected.mean() * abs(confidence[selected].mean() - right[selected].mean()))
    result = {
        "samples": len(rows), "cause_labels": list(LABELS),
        "cause_accuracy": float(accuracy_score(cause_true, cause_pred)),
        "cause_macro_f1": float(f1_score(cause_true, cause_pred, labels=range(len(LABELS)), average="macro", zero_division=0)),
        "cause_per_class": {label: {"precision": float(precision[i]), "recall": float(recall[i]),
                                    "f1": float(f1[i]), "support": int(support[i])} for i, label in enumerate(LABELS)},
        "cause_confusion_matrix": confusion_matrix(cause_true, cause_pred, labels=range(len(LABELS))).tolist(),
        # Error attribution: given a true error, was the exact cause predicted?
        # Missed errors count as wrong in the first metric and are excluded in the second.
        "error_type_accuracy": (float((cause_pred[error_mask] == cause_true[error_mask]).mean())
                                if error_support else None),
        "error_type_support": error_support,
        "error_type_accuracy_when_detected": (float((cause_pred[detected_mask] == cause_true[detected_mask]).mean())
                                              if detected_support else None),
        "error_type_detected_support": detected_support,
        "correctness_accuracy": float(accuracy_score(correct_true, correct_pred)),
        "error_detection_f1": float(f1_score(wrong_true, wrong_pred, zero_division=0)),
        "error_auroc": float(roc_auc_score(wrong_true, 1 - correct_probs)) if correct_count and wrong_count else None,
        "false_alarm_rate": float((correct_true & wrong_pred).sum() / correct_count) if correct_count else None,
        "missed_error_rate": float((wrong_true & correct_pred).sum() / wrong_count) if wrong_count else None,
        "correctness_brier": float(((correct_probs - correct_true) ** 2).mean()),
        "correctness_ece": ece,
        "inconsistent_answers": int(((cause_pred == 0) != correct_pred).sum()),
    }
    if latencies_ms:
        result["latency_ms"] = {"median": float(np.median(latencies_ms)), "p95": float(np.percentile(latencies_ms, 95)),
                                "measurement": "one tool-call example per request, warm model, includes preprocessing"}
    return result
