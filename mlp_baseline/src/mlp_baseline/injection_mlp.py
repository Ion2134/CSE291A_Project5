from __future__ import annotations

import csv
import hashlib
import json
import platform
import site
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support, roc_auc_score)

from .data import input_id, write_json, write_jsonl
from .injection_data import INJECTION_LABELS, INJECTION_QUESTIONS
from .metrics import fit_temperature, probabilities, scale_temperature
from .mlp import load_mlp, train_mlp


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_injection_splits(data: Path):
    manifest = json.loads((data / "manifest.json").read_text())
    if manifest["task"] != "injection_detection":
        raise ValueError("Expected a prepared injection-detection dataset")
    splits = {name: _read(data / f"{name}.jsonl") for name in ("train", "val", "test")}
    for name, rows in splits.items():
        if hashlib.sha256((data / f"{name}.jsonl").read_bytes()).hexdigest() != manifest["split_sha256"][name]:
            raise ValueError(f"{name} data hash differs from its preparation manifest")
        if not rows or any(row["questions"] != INJECTION_QUESTIONS or row["expected"].get("injection") not in INJECTION_LABELS for row in rows):
            raise ValueError(f"Invalid {name} injection schema or labels")
        if {row["expected"]["injection"] for row in rows} != set(INJECTION_LABELS):
            raise ValueError(f"{name} must contain both classes")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if {input_id(row) for row in splits[a]} & {input_id(row) for row in splits[b]}:
            raise ValueError("Identical input leaked across injection splits")
        if {row["group_id"] for row in splits[a]} & {row["group_id"] for row in splits[b]}:
            raise ValueError("Similarity group leaked across injection splits")
    return splits, manifest


def binary_metrics(rows, values):
    values = probabilities(values)
    if values.shape != (len(rows), 2):
        raise ValueError("Expected two probabilities per injection example")
    actual = np.array([INJECTION_LABELS.index(row["expected"]["injection"]) for row in rows])
    predicted = values.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(actual, predicted, labels=[0, 1], zero_division=0)
    negative, positive = actual == 0, actual == 1
    return {
        "samples": len(rows), "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=[0, 1], average="macro", zero_division=0)),
        "injection_precision": float(precision[1]), "injection_recall": float(recall[1]), "injection_f1": float(f1[1]),
        "auroc": float(roc_auc_score(actual, values[:, 1])) if negative.any() and positive.any() else None,
        "average_precision": float(average_precision_score(actual, values[:, 1])) if positive.any() else None,
        "false_alarm_rate": float((predicted[negative] == 1).mean()) if negative.any() else None,
        "missed_injection_rate": float((predicted[positive] == 0).mean()) if positive.any() else None,
        "brier": float(((values[:, 1] - actual) ** 2).mean()),
        "log_loss": float(-np.log(np.clip(values[np.arange(len(rows)), actual], 1e-9, 1)).mean()),
        "labels": list(INJECTION_LABELS), "confusion_matrix": confusion_matrix(actual, predicted, labels=[0, 1]).tolist(),
        "per_class": {label: {"precision": float(precision[i]), "recall": float(recall[i]),
                               "f1": float(f1[i]), "support": int(support[i])} for i, label in enumerate(INJECTION_LABELS)},
    }


class InjectionMLPPredictor:
    def __init__(self, predictor, calibration):
        self.predictor, self.calibration = predictor, calibration

    def predict_proba(self, rows):
        return scale_temperature(self.predictor.predict_proba(rows), self.calibration["temperature"])


def load_injection_mlp(path: Path):
    metadata = json.loads((path / "monitor.json").read_text())
    if metadata["task"] != "injection_detection" or metadata["labels"] != list(INJECTION_LABELS):
        raise ValueError("Checkpoint is not an injection-detection MLP")
    return InjectionMLPPredictor(load_mlp(path, "cpu"), metadata["calibration"])


def _predictions(rows, values):
    return [{"input_id": input_id(row), "source": row.get("meta", {}).get("source"),
             "source_id": row.get("meta", {}).get("source_id"),
             "expected": row.get("expected"), "prediction": INJECTION_LABELS[int(prob.argmax())],
             "injection_probability": float(prob[1])} for row, prob in zip(rows, values)]


def train_injection_mlp(data: Path, out: Path, *, seed=42, epochs=40, batch_size=64,
                        max_features=4096, patience=8, threads=4):
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new or empty MLP experiment directory")
    if sys.prefix == sys.base_prefix or site.ENABLE_USER_SITE:
        raise RuntimeError("Run MLP training inside the isolated project venv")
    torch.set_num_threads(threads)
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    result = {"status": "running", "task": "injection_detection", "model": "MLP", "device": "cpu",
              "isolated_venv": True, "python": platform.python_version(), "seed": seed,
              "data_path": str(data), "data_manifest_sha256": hashlib.sha256((data / "manifest.json").read_bytes()).hexdigest(),
              "scope": "Public-message injection detection. This does not measure banking tool-call correctness or error-cause attribution."}
    write_json(out / "results.json", result)
    try:
        splits, manifest = validate_injection_splits(data)
        result["data"] = manifest
        result["phase"] = "training"
        write_json(out / "results.json", result)
        print("Training local MLP: " + ", ".join(f"{name}={len(rows)}" for name, rows in splits.items()), flush=True)
        model_dir = out / "mlp"
        details = train_mlp(splits["train"], splits["val"], model_dir,
                            seed=seed, epochs=epochs, batch_size=batch_size, max_features=max_features,
                            labels=INJECTION_LABELS, target="injection", patience=patience, device="cpu", verbose=True)
        raw = load_mlp(model_dir, "cpu")
        val_probs = raw.predict_proba(splits["val"])
        calibration = fit_temperature(val_probs, [INJECTION_LABELS.index(row["expected"]["injection"]) for row in splits["val"]])
        calibration["source"] = "validation only"
        write_json(model_dir / "monitor.json", {"kind": "mlp", "task": "injection_detection",
                                                "labels": list(INJECTION_LABELS), "calibration": calibration,
                                                "input_fields": ["state"]})
        monitor = load_injection_mlp(model_dir)
        probabilities_test = monitor.predict_proba(splits["test"])
        metrics = binary_metrics(splits["test"], probabilities_test)
        write_jsonl(out / "test_predictions.jsonl", _predictions(splits["test"], probabilities_test))
        source_metrics = {}
        for source in sorted({row["meta"]["source"] for row in splits["test"]}):
            selected = [i for i, row in enumerate(splits["test"]) if row["meta"]["source"] == source]
            source_metrics[source] = binary_metrics([splits["test"][i] for i in selected], probabilities_test[selected])
        # Measure warm single-message latency separately from the batched scoring pass.
        monitor.predict_proba(splits["test"][:1])
        latencies = []
        for row in splits["test"][:100]:
            stamp = time.perf_counter()
            monitor.predict_proba([row])
            latencies.append((time.perf_counter() - stamp) * 1000)
        metrics["latency_ms"] = {"median": float(np.median(latencies)), "p95": float(np.percentile(latencies, 95)), "samples": len(latencies)}
        # Verify the complete deployed artifact, including vocabulary and calibration.
        restored = load_injection_mlp(model_dir).predict_proba(splits["test"])
        np.testing.assert_allclose(probabilities_test, restored, atol=1e-6, rtol=1e-6)
        details["checkpoint_reload_verified"] = True
        majority = Counter(row["expected"]["injection"] for row in splits["train"]).most_common(1)[0][0]
        majority_probs = np.tile(np.eye(2)[INJECTION_LABELS.index(majority)], (len(splits["test"]), 1))
        result.update(training=details, calibration=calibration, test=metrics, test_by_source=source_metrics,
                      majority_baseline=binary_metrics(splits["test"], majority_probs))
        diagnostic_dir = data / "diagnostics"
        result["diagnostics"] = {}
        for path in sorted(diagnostic_dir.glob("toolfault-*.jsonl")):
            rows = _read(path)
            values = monitor.predict_proba(rows)
            result["diagnostics"][path.stem] = binary_metrics(rows, values)
            write_jsonl(out / f"{path.stem}_predictions.jsonl", _predictions(rows, values))
        result.update(status="complete", phase="complete", elapsed_seconds=time.perf_counter() - started)
        write_json(out / "results.json", result)
        columns = ["dataset", "samples", "accuracy", "macro_f1", "injection_precision", "injection_recall", "injection_f1", "auroc", "false_alarm_rate", "missed_injection_rate"]
        with (out / "metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for name, value in {"test_combined": metrics, **source_metrics, **result["diagnostics"]}.items():
                writer.writerow({"dataset": name, **{key: value[key] for key in columns[1:]}})
        print(json.dumps({"status": "complete", "parameters": details["parameters"], "test_macro_f1": metrics["macro_f1"], "results": str(out / "results.json")}), flush=True)
        return result
    except Exception as exc:
        result.update(status="failed", phase="failed", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.perf_counter() - started)
        write_json(out / "results.json", result)
        raise
