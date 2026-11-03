"""Validation loss, temperature calibration, and test metrics for a fine-tuned Laya checkpoint."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import LABELS

# Question types with fewer validation items (or a single target class) keep temperature 1.
MIN_CALIBRATION_ITEMS = 20
PROBABILITY_FLOOR = 1e-7


def validation_loss(records) -> float:
    """Mean soft cross-entropy of Laya's calibration records ``(type, logits, target, ...)``."""
    losses = []
    for _, logits, target, _ in records:
        values = np.asarray(logits, dtype=np.float64)
        log_probs = values - np.max(values)
        log_probs -= np.log(np.exp(log_probs).sum())
        losses.append(float(-np.dot(target, log_probs)))

    result = float(np.mean(losses))
    if not math.isfinite(result):
        raise ValueError("Non-finite validation loss; inspect the training log")
    return result


def fit_calibration(records, minimum=MIN_CALIBRATION_ITEMS) -> dict:
    """Fit temperatures on validation only; small or single-class question types stay at 1."""
    from laya.calibrate import fit_temperature_map

    by_type = defaultdict(list)
    for record in records:
        by_type[int(record[0])].append(record)

    eligible, skipped = [], {}
    for question_type, part in by_type.items():
        if len(part) < minimum or len({int(np.argmax(r[2])) for r in part}) < 2:
            skipped[str(question_type)] = {"n": len(part),
                                           "reason": "need >=20 validation items and >=2 target classes"}
        else:
            eligible.extend(part)

    fitted = fit_temperature_map(eligible)
    return {**fitted, "skipped_types": skipped, "source": "validation only", "minimum_items": minimum}


# ---------------------------------------------------------------- test metrics

def evaluate(agent, rows, output_file: Path, batch_size: int) -> dict:
    """Predict every row, save the raw answers to ``output_file``, and score each question."""
    answers = _predict(agent, rows, batch_size)
    with Path(output_file).open("w", encoding="utf-8") as handle:
        for index, (row, answer) in enumerate(zip(rows, answers)):
            handle.write(json.dumps({"row": index, "meta": row.get("meta"), "expected": row["expected"],
                                     "answers": answer}) + "\n")

    metrics = {qid: _question_metrics(qid, rows, answers) for qid in rows[0]["questions"]}
    if "injection" in metrics:
        # A per-source breakdown exposes the domain gap an aggregate score hides.
        metrics["by_source"] = _injection_by_source(rows, answers)
    return metrics


def _predict(agent, rows, batch_size: int) -> list[dict]:
    answers = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        outputs = agent.predict_batch([r["state"] for r in chunk], chunk[0]["questions"], batch_size=batch_size)
        answers.extend(output["answers"] for output in outputs)
    return answers


def _question_metrics(qid: str, rows, answers) -> dict:
    """Accuracy, F1, confusion matrix and log loss; binary questions add AUROC, Brier and error rates."""
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, roc_auc_score

    labels = LABELS[qid]
    indices = list(range(len(labels)))
    y = np.array([labels.index(r["expected"][qid]) for r in rows])
    p = _answer_probabilities(qid, labels, answers)
    pred = p.argmax(axis=1)

    result = {"rows": len(y), "labels": labels, "accuracy": float(accuracy_score(y, pred)),
              "macro_f1": float(f1_score(y, pred, labels=indices, average="macro", zero_division=0)),
              "per_class_f1": f1_score(y, pred, labels=indices, average=None, zero_division=0).tolist(),
              "class_support": np.bincount(y, minlength=len(labels)).tolist(),
              "confusion_matrix": confusion_matrix(y, pred, labels=indices).tolist(),
              "log_loss": float(log_loss(y, p, labels=indices))}
    if len(labels) == 2:
        result.update(auroc=float(roc_auc_score(y, p[:, 1])) if len(set(y)) == 2 else None,
                      brier=float(np.mean((p[:, 1] - y) ** 2)),
                      false_positive_rate=float(np.mean(pred[y == 0] == 1)) if np.any(y == 0) else None,
                      false_negative_rate=float(np.mean(pred[y == 1] == 0)) if np.any(y == 1) else None)
    return result


def _answer_probabilities(qid: str, labels, answers) -> np.ndarray:
    """Laya's answers as a row-normalized probability matrix in ``labels`` order."""
    if qid == "output_correct":
        p = np.array([[1 - a[qid]["noul"], a[qid]["noul"]] for a in answers])
    else:
        p = np.array([[a[qid]["probabilities"][label] for label in labels] for a in answers])
    p = np.clip(p, PROBABILITY_FLOOR, 1)
    p /= p.sum(axis=1, keepdims=True)
    return p


def _injection_by_source(rows, answers) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    def source_of(row):
        return row.get("meta", {}).get("source", "unknown")

    by_source = {}
    for source in sorted({source_of(r) for r in rows}):
        indices = [i for i, r in enumerate(rows) if source_of(r) == source]
        y = [LABELS["injection"].index(rows[i]["expected"]["injection"]) for i in indices]
        probabilities = [answers[i]["injection"]["probabilities"] for i in indices]
        pred = [int(p["injection"] > p["benign"]) for p in probabilities]
        by_source[source] = {"rows": len(y), "accuracy": float(accuracy_score(y, pred)),
                             "macro_f1": float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0))}
    return by_source
