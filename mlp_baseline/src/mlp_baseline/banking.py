from __future__ import annotations

import gc
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .banking_data import prepare_training_data
from .data import LABELS, input_id, read_rows, split_rows, write_json, write_jsonl
from .metrics import evaluate, fit_temperature, scale_temperature
from .mlp import load_mlp, train_mlp


class BankingMLPPredictor:
    def __init__(self, predictor, calibration):
        self.predictor, self.calibration = predictor, calibration

    def predict(self, rows):
        cause, correct = self.predictor.predict(rows)
        cause = scale_temperature(cause, self.calibration["cause"]["temperature"])
        binary = scale_temperature(np.stack([1 - correct, correct], axis=1),
                                   self.calibration["correctness"]["temperature"])
        return cause, binary[:, 1]


def load_banking_mlp(path: Path):
    metadata = json.loads((path / "monitor.json").read_text())
    if metadata["labels"] != list(LABELS):
        raise ValueError("Expected a banking cause-classification checkpoint")
    return BankingMLPPredictor(load_mlp(path, "cpu"), metadata["calibration"])


def run_banking(run: Path, out: Path, *, seed=42, epochs=40, batch_size=16, threads=4):
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use an empty banking experiment directory")
    torch.set_num_threads(threads)
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    audit = prepare_training_data(run, out / "prepared")
    rows, equivalent_duplicates = read_rows(out / "prepared/laya_train.jsonl")
    splits, manifest = split_rows(rows, seed)
    split_dir = out / "splits"
    split_dir.mkdir()
    for name, part in splits.items():
        write_jsonl(split_dir / f"{name}.jsonl", part)
    manifest.update(audit=audit, equivalent_duplicates_removed=equivalent_duplicates)
    write_json(split_dir / "manifest.json", manifest)
    train, val, test = splits["train"], splits["val"], splits["test"]
    model_dir = out / "mlp"
    details = train_mlp(train, val, model_dir, seed=seed, epochs=epochs,
                        batch_size=batch_size, device="cpu", verbose=True)
    predictor = load_mlp(model_dir)
    cause, correct = predictor.predict(val)
    calibration = {
        "cause": fit_temperature(cause, [LABELS.index(row["expected"]["cause"]) for row in val]),
        "correctness": fit_temperature(np.stack([1 - correct, correct], axis=1),
                                       [int(row["expected"]["output_correct"]) for row in val]),
        "source": "validation only",
    }
    write_json(model_dir / "monitor.json", {"kind": "mlp", "task": "banking_monitor",
                                            "labels": list(LABELS), "calibration": calibration})
    del predictor
    gc.collect()
    monitor = load_banking_mlp(model_dir)
    monitor.predict(test[:1])
    causes, corrects, latencies, predictions = [], [], [], []
    for row in test:
        stamp = time.perf_counter()
        cause, correct = monitor.predict([row])
        latency = (time.perf_counter() - stamp) * 1000
        causes.append(cause[0])
        corrects.append(float(correct[0]))
        latencies.append(latency)
        predictions.append({"input_id": input_id(row), "expected": row["expected"],
                            "cause_probabilities": dict(zip(LABELS, map(float, cause[0]))),
                            "correct_probability": float(correct[0]), "latency_ms": latency})
    metrics = evaluate(test, causes, corrects, latencies)
    restored = load_banking_mlp(model_dir).predict(test)
    np.testing.assert_allclose(restored[0], causes, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(restored[1], corrects, atol=1e-6, rtol=1e-6)
    details["checkpoint_reload_verified"] = True
    majority = Counter(row["expected"]["cause"] for row in train).most_common(1)[0][0]
    majority_probs = np.tile(np.eye(3)[LABELS.index(majority)], (len(test), 1))
    result = {"status": "complete", "task": "banking_monitor", "development_only": True,
              "device": "cpu", "seed": seed, "data": manifest, "training": details,
              "test": metrics, "calibration": calibration,
              "majority_baseline": evaluate(test, majority_probs, majority_probs[:, 0]),
              "elapsed_seconds": time.perf_counter() - started,
              "limitation": "Only 52 distinct banking cases and 11 test examples; this is a development result."}
    write_jsonl(out / "test_predictions.jsonl", predictions)
    write_json(out / "results.json", result)
    return result
