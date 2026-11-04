"""``mlp-baseline train-injection``: train the injection detector and write its reports.

Train on ``train``, keep the epoch with the lowest validation loss, fit a temperature on
validation, then score test (overall and per source) and the ToolFault diagnostics.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .common import (RunRecord, read_jsonl, require_empty_dir, require_isolated_venv, sha256_file, write_json,
                     write_jsonl)
from .injection_data import INJECTION_LABELS
from .injection_model import (TASK, InjectionMLPPredictor, binary_metrics, load_injection_mlp, prediction_rows,
                              validate_injection_splits)
from .metrics import fit_temperature, majority_probabilities
from .mlp import MONITOR_FILE, TrainSettings, load_mlp, train_mlp

# The injection detector's defaults (``mlp-baseline train-injection``).
DEFAULT_SETTINGS = TrainSettings(batch_size=64, patience=8)

DIAGNOSTICS_DIR = "diagnostics"
LATENCY_SAMPLES = 100
SCOPE = ("Public-message injection detection. This does not measure banking tool-call correctness or "
         "error-cause attribution.")
METRICS_CSV_COLUMNS = ["dataset", "samples", "accuracy", "macro_f1", "injection_precision", "injection_recall",
                       "injection_f1", "auroc", "false_alarm_rate", "missed_injection_rate"]


@dataclass(frozen=True)
class InjectionRunFiles:
    """Everything one training run writes under ``out``."""

    out: Path

    @property
    def results(self) -> Path:
        return self.out / "results.json"

    @property
    def model(self) -> Path:
        return self.out / "mlp"

    @property
    def test_predictions(self) -> Path:
        return self.out / "test_predictions.jsonl"

    def diagnostic_predictions(self, name: str) -> Path:
        return self.out / f"{name}_predictions.jsonl"

    @property
    def metrics_csv(self) -> Path:
        return self.out / "metrics.csv"


@dataclass
class TestScores:
    """The calibrated test probabilities and their metrics, overall and per source."""

    probabilities: np.ndarray
    metrics: dict
    by_source: dict


def train_injection_mlp(data: Path, out: Path, settings: TrainSettings = DEFAULT_SETTINGS) -> dict:
    """Train on ``data``, calibrate on validation, score test and ToolFault, write the reports."""
    files = InjectionRunFiles(out)
    require_empty_dir(out, "Use a new or empty MLP experiment directory")
    require_isolated_venv()
    torch.set_num_threads(settings.threads)
    out.mkdir(parents=True, exist_ok=True)

    with RunRecord(files.results, _run_header(data, settings.seed)) as record:
        splits, manifest = validate_injection_splits(data)
        record.update(data=manifest)
        record.set_phase("training")
        print("Training local MLP: " + ", ".join(f"{name}={len(rows)}" for name, rows in splits.items()), flush=True)

        details, calibration = _train_and_calibrate(splits, files.model, settings)
        monitor = load_injection_mlp(files.model)

        test = _score_test(monitor, splits["test"], files)
        _verify_reload(files.model, splits["test"], test.probabilities)
        details["checkpoint_reload_verified"] = True

        record.update(training=details, calibration=calibration, test=test.metrics, test_by_source=test.by_source,
                      majority_baseline=_majority_baseline(splits["train"], splits["test"]),
                      diagnostics=_score_diagnostics(monitor, data / DIAGNOSTICS_DIR, files))

    result = record.result
    _write_metrics_csv(files.metrics_csv, {"test_combined": test.metrics, **test.by_source, **result["diagnostics"]})
    print(json.dumps({"status": "complete", "parameters": details["parameters"],
                      "test_macro_f1": test.metrics["macro_f1"], "results": str(files.results)}), flush=True)
    return result


def _run_header(data: Path, seed: int) -> dict:
    return {"status": "running", "task": TASK, "model": "MLP", "device": "cpu", "isolated_venv": True,
            "python": platform.python_version(), "seed": seed, "data_path": str(data),
            "data_manifest_sha256": sha256_file(data / "manifest.json"), "scope": SCOPE}


def _train_and_calibrate(splits, model_dir: Path, settings: TrainSettings):
    """Train the MLP, fit its temperature on validation only, and save the inference metadata."""
    details = train_mlp(splits["train"], splits["val"], model_dir, dataclasses.replace(settings, verbose=True),
                        labels=INJECTION_LABELS, target="injection", device="cpu")

    val_probs = load_mlp(model_dir, "cpu").predict_proba(splits["val"])
    val_targets = [INJECTION_LABELS.index(row["expected"]["injection"]) for row in splits["val"]]
    calibration = fit_temperature(val_probs, val_targets)
    calibration["source"] = "validation only"

    write_json(model_dir / MONITOR_FILE, {"kind": "mlp", "task": TASK, "labels": list(INJECTION_LABELS),
                                          "calibration": calibration, "input_fields": ["state"]})
    return details, calibration


def _score_test(monitor: InjectionMLPPredictor, test, files: InjectionRunFiles) -> TestScores:
    """Test probabilities and metrics (with warm single-message latency), overall and per source."""
    test_probs = monitor.predict_proba(test)
    metrics = binary_metrics(test, test_probs)
    write_jsonl(files.test_predictions, prediction_rows(test, test_probs))

    by_source = {}
    for source in sorted({row["meta"]["source"] for row in test}):
        selected = [i for i, row in enumerate(test) if row["meta"]["source"] == source]
        by_source[source] = binary_metrics([test[i] for i in selected], test_probs[selected])

    metrics["latency_ms"] = _single_message_latency(monitor, test)
    return TestScores(test_probs, metrics, by_source)


def _single_message_latency(monitor: InjectionMLPPredictor, rows) -> dict:
    """Warm latency of predicting one message at a time, separate from the batched scoring pass."""
    monitor.predict_proba(rows[:1])
    latencies = []
    for row in rows[:LATENCY_SAMPLES]:
        started = time.perf_counter()
        monitor.predict_proba([row])
        latencies.append((time.perf_counter() - started) * 1000)
    return {"median": float(np.median(latencies)), "p95": float(np.percentile(latencies, 95)),
            "samples": len(latencies)}


def _verify_reload(model_dir: Path, test, test_probs) -> None:
    """The saved artifact (weights, vocabulary, calibration) must reproduce the test probabilities."""
    restored = load_injection_mlp(model_dir).predict_proba(test)
    np.testing.assert_allclose(test_probs, restored, atol=1e-6, rtol=1e-6)


def _majority_baseline(train, test) -> dict:
    probs = majority_probabilities([row["expected"]["injection"] for row in train], len(test), INJECTION_LABELS)
    return binary_metrics(test, probs)


def _score_diagnostics(monitor: InjectionMLPPredictor, diagnostic_dir: Path, files: InjectionRunFiles) -> dict:
    """ToolFault files are evaluation-only: score each and save its predictions."""
    diagnostics = {}
    for path in sorted(diagnostic_dir.glob("toolfault-*.jsonl")):
        rows = read_jsonl(path)
        values = monitor.predict_proba(rows)
        diagnostics[path.stem] = binary_metrics(rows, values)
        write_jsonl(files.diagnostic_predictions(path.stem), prediction_rows(rows, values))
    return diagnostics


def _write_metrics_csv(path: Path, by_dataset: dict) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRICS_CSV_COLUMNS)
        writer.writeheader()
        for name, metrics in by_dataset.items():
            writer.writerow({"dataset": name, **{key: metrics[key] for key in METRICS_CSV_COLUMNS[1:]}})
