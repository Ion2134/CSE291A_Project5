"""The trained injection detector: load a checkpoint, predict, and score benign-vs-injection predictions."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support, roc_auc_score)

from .common import input_id, read_json, read_jsonl, sha256_file
from .injection_data import INJECTION_LABELS, INJECTION_QUESTIONS, SPLITS
from .metrics import probabilities, scale_temperature
from .mlp import MONITOR_FILE, MLPPredictor, load_mlp

TASK = "injection_detection"
INJECTION_INDEX = INJECTION_LABELS.index("injection")


@dataclass
class InjectionMLPPredictor:
    """A trained injection MLP plus its validation temperature."""

    predictor: MLPPredictor
    calibration: dict

    def predict_proba(self, rows):
        """``[P(benign), P(injection)]`` per row, calibrated."""
        return scale_temperature(self.predictor.predict_proba(rows), self.calibration["temperature"])


def load_injection_mlp(path: Path) -> InjectionMLPPredictor:
    metadata = read_json(path / MONITOR_FILE)
    if metadata["task"] != TASK or metadata["labels"] != list(INJECTION_LABELS):
        raise ValueError("Checkpoint is not an injection-detection MLP")
    return InjectionMLPPredictor(load_mlp(path, "cpu"), metadata["calibration"])


def prediction_rows(rows, values) -> list[dict]:
    """One output row per example: its ids, label, predicted class and P(injection)."""
    return [{"input_id": input_id(row), "source": row.get("meta", {}).get("source"),
             "source_id": row.get("meta", {}).get("source_id"),
             "expected": row.get("expected"), "prediction": INJECTION_LABELS[int(prob.argmax())],
             "injection_probability": float(prob[INJECTION_INDEX])}
            for row, prob in zip(rows, values)]


def validate_injection_splits(data: Path):
    """Load a prepared injection dataset and check hashes, schema, classes and split isolation."""
    manifest = read_json(data / "manifest.json")
    if manifest["task"] != TASK:
        raise ValueError("Expected a prepared injection-detection dataset")

    splits = {name: read_jsonl(data / f"{name}.jsonl") for name in SPLITS}
    for name, rows in splits.items():
        if sha256_file(data / f"{name}.jsonl") != manifest["split_sha256"][name]:
            raise ValueError(f"{name} data hash differs from its preparation manifest")
        if not rows or any(row["questions"] != INJECTION_QUESTIONS
                           or row["expected"].get("injection") not in INJECTION_LABELS for row in rows):
            raise ValueError(f"Invalid {name} injection schema or labels")
        if {row["expected"]["injection"] for row in rows} != set(INJECTION_LABELS):
            raise ValueError(f"{name} must contain both classes")

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if {input_id(row) for row in splits[a]} & {input_id(row) for row in splits[b]}:
            raise ValueError("Identical input leaked across injection splits")
        if {row["group_id"] for row in splits[a]} & {row["group_id"] for row in splits[b]}:
            raise ValueError("Similarity group leaked across injection splits")
    return splits, manifest


def binary_metrics(rows, values) -> dict:
    """Accuracy, F1, AUROC, false-alarm and miss rates, Brier and log loss for benign/injection."""
    values = probabilities(values)
    if values.shape != (len(rows), 2):
        raise ValueError("Expected two probabilities per injection example")

    actual = np.array([INJECTION_LABELS.index(row["expected"]["injection"]) for row in rows])
    predicted = values.argmax(axis=1)
    p_injection = values[:, INJECTION_INDEX]
    negative, positive = actual == 0, actual == 1
    precision, recall, f1, support = precision_recall_fscore_support(actual, predicted, labels=[0, 1],
                                                                     zero_division=0)
    return {
        "samples": len(rows),
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=[0, 1], average="macro", zero_division=0)),
        "injection_precision": float(precision[1]),
        "injection_recall": float(recall[1]),
        "injection_f1": float(f1[1]),
        "auroc": float(roc_auc_score(actual, p_injection)) if negative.any() and positive.any() else None,
        "average_precision": float(average_precision_score(actual, p_injection)) if positive.any() else None,
        "false_alarm_rate": float((predicted[negative] == 1).mean()) if negative.any() else None,
        "missed_injection_rate": float((predicted[positive] == 0).mean()) if positive.any() else None,
        "brier": float(((p_injection - actual) ** 2).mean()),
        "log_loss": float(-np.log(np.clip(values[np.arange(len(rows)), actual], 1e-9, 1)).mean()),
        "labels": list(INJECTION_LABELS),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=[0, 1]).tolist(),
        "per_class": {label: {"precision": float(precision[i]), "recall": float(recall[i]),
                              "f1": float(f1[i]), "support": int(support[i])}
                      for i, label in enumerate(INJECTION_LABELS)},
    }
