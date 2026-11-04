"""The early banking development experiment (archived): one simulator run, one MLP.

``prepare_training_data`` repairs and deduplicates one run's ``laya_train.jsonl``;
``run_banking`` splits it by request group, trains a three-class MLP, calibrates on validation
and scores the test split. Superseded by ``agent_runs`` + ``monitor_training`` and the unified
tester in ``testing/``; kept so ``prepare-banking`` / ``train-banking`` still reproduce
``archive/banking-dev``.
"""
from __future__ import annotations

import gc
import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from .agent_runs import EXAMPLES_FILE, LAYA_ROWS_FILE, REPAIRED_IO_REASON, repair_noop_io_label
from .common import (LABELS, MONITOR_QUESTIONS, cause_counts, cause_of, input_id, read_json, read_jsonl,
                     request_group, require_empty_dir, sha256_file, valid_monitor_labels, write_json, write_jsonl)
from .metrics import as_binary, evaluate, fit_calibration, majority_probabilities, scale_temperature
from .mlp import MONITOR_FILE, MLPPredictor, TrainSettings, load_mlp, train_mlp

SPLIT_ATTEMPTS = 1000
MISSING_HOLDOUT_CLASS_PENALTY = 10
LIMITATION = "Only 52 distinct banking cases and 11 test examples; this is a development result."
BASELINE_NOTE = ("Raw run files and baseline.json are historical and unchanged; baseline metrics are not "
                 "recomputed here.")


# ---------------------------------------------------------------- preparing one run

def prepare_training_data(run_dir: Path, out_dir: Path | None = None) -> dict:
    """Write a repaired, deduplicated copy of one run plus an audit trail; the raw run stays intact."""
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir is not None else run_dir / "prepared"
    if out_dir.resolve() == run_dir.resolve():
        raise ValueError("Use a separate output directory to preserve the raw run files")

    source_path = run_dir / LAYA_ROWS_FILE
    rows = read_jsonl(source_path)
    if not rows:
        raise ValueError("No training rows found")
    examples_path = run_dir / EXAMPLES_FILE
    examples = read_jsonl(examples_path) if examples_path.exists() else None
    if examples is not None and len(examples) != len(rows):
        raise ValueError("examples.jsonl and laya_train.jsonl have different row counts")

    cleaned, provenance, corrections = _repair_and_deduplicate(rows, examples)

    report = {
        "source_file": source_path.name,
        "source_sha256": sha256_file(source_path),
        "examples_sha256": sha256_file(examples_path) if examples is not None else None,
        "raw_rows": len(rows),
        "repaired_rows": len(corrections),
        "duplicate_rows_removed": len(rows) - len(cleaned),
        "unique_inputs": len(cleaned),
        "label_counts": cause_counts(cleaned),
        "remaining_conflicting_inputs": 0,
        "label_corrections": corrections,
        "baseline_note": BASELINE_NOTE,
    }
    # Everything is validated before anything is written, so a conflict never leaves partial output.
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / LAYA_ROWS_FILE, cleaned)
    write_jsonl(out_dir / "provenance.jsonl", provenance)
    write_json(out_dir / "data_report.json", report)
    return report


def _repair_and_deduplicate(rows, examples):
    """Repair each row (when ``examples`` are available) and drop exact duplicate inputs.

    Expected answers are not part of the duplicate key: different answers to the same input are a
    data error, not separate examples to keep or vote on.
    """
    cleaned, provenance, corrections = [], [], []
    seen: dict[str, int] = {}
    for line, original in enumerate(rows, 1):
        row = original
        example = examples[line - 1] if examples is not None else None
        if example is not None:
            expected = {key: example["label"][key] for key in ("output_correct", "cause")}
            if example["input"]["state_text"] != original["state"] or expected != original["expected"]:
                raise ValueError(f"Source files disagree at row {line}; no output written")
            row = repair_noop_io_label(original, example)
            if row != original:
                corrections.append({"source_line": line, "example_id": example["example_id"],
                                    "before": original["expected"], "after": row["expected"],
                                    "reason": REPAIRED_IO_REASON})

        key = json.dumps({"state": row["state"], "questions": row["questions"]}, sort_keys=True)
        source = {"source_line": line}
        if example is not None:
            source["example_id"] = example["example_id"]

        if key not in seen:
            seen[key] = len(cleaned)
            cleaned.append(row)
            provenance.append({"clean_line": len(cleaned), "sources": [source]})
            continue
        index = seen[key]
        if cleaned[index]["expected"] != row["expected"]:
            first = provenance[index]["sources"][0]["source_line"]
            raise ValueError(f"Unresolved conflicting labels at rows {first} and {line}; no output written")
        provenance[index]["sources"].append(source)
    return cleaned, provenance, corrections


def read_rows(path: Path) -> tuple[list[dict], int]:
    """Validated, deduplicated rows of one prepared file, plus how many duplicates were dropped."""
    rows = read_jsonl(path)
    if not rows:
        raise ValueError("Training data is empty")

    seen = {}
    for index, row in enumerate(rows, 1):
        if not valid_monitor_labels(row["expected"]):
            raise ValueError(f"Invalid or inconsistent monitor labels at row {index}")
        if row["questions"] != rows[0]["questions"]:
            raise ValueError("All rows must use the same monitor question schema")
        if set(row["questions"]) != MONITOR_QUESTIONS:
            raise ValueError("Expected output_correct and cause questions")
        key = input_id(row)
        if key in seen and seen[key]["expected"] != row["expected"]:
            raise ValueError(f"Conflicting labels for equivalent inputs at row {index}")
        seen.setdefault(key, row)
    return list(seen.values()), len(rows) - len(seen)


# ---------------------------------------------------------------- the dev experiment's own split

def split_rows(rows: list[dict], seed: int = 42, val_fraction: float = 0.2,
               test_fraction: float = 0.2) -> tuple[dict[str, list[dict]], dict]:
    """Train/val/test by whole request groups; training must keep every cause."""
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and val_fraction + test_fraction < 1):
        raise ValueError("Validation/test fractions must be positive and sum to less than 1")
    groups = defaultdict(list)
    for row in rows:
        groups[request_group(row)].append(row)
    keys = sorted(groups)
    if len(keys) < 3:
        raise ValueError("Need at least three independent user-request groups for train/val/test")
    n_val = max(1, round(len(keys) * val_fraction))
    n_test = max(1, round(len(keys) * test_fraction))
    if n_val + n_test >= len(keys):
        n_val = n_test = 1
    total = Counter(cause_of(row) for row in rows)
    if set(total) != set(LABELS):
        raise ValueError("The dataset must include correct, agent-error, and adversary-error examples")

    fractions = {"train": 1 - val_fraction - test_fraction, "val": val_fraction, "test": test_fraction}
    rng = random.Random(seed)
    best = None
    for _ in range(SPLIT_ATTEMPTS):
        order = keys.copy()
        rng.shuffle(order)
        partitions = {"val": order[:n_val], "test": order[n_val:n_val + n_test], "train": order[n_val + n_test:]}
        candidate = {name: [row for key in selected for row in groups[key]] for name, selected in partitions.items()}
        counts = {name: Counter(cause_of(row) for row in part) for name, part in candidate.items()}
        if set(counts["train"]) != set(LABELS):
            continue
        score = _split_score(candidate, counts, total, fractions, len(rows))
        if best is None or score < best[0]:
            best = (score, candidate, partitions)
    if best is None:
        raise ValueError("Cannot keep all classes in training with independent holdouts; generate more request groups")

    _, splits, partitions = best
    manifest = {
        "seed": seed, "strategy": "group_by_normalized_user_request",
        "fractions_requested": {"val": val_fraction, "test": test_fraction},
        "splits": {name: {"rows": len(part), "groups": partitions[name],
                          "input_ids": [input_id(row) for row in part],
                          "label_counts": dict(Counter(cause_of(row) for row in part))}
                   for name, part in splits.items()},
        "warnings": _split_warnings(total, splits),
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not set(manifest["splits"][left]["input_ids"]) & set(manifest["splits"][right]["input_ids"])
        assert not set(partitions[left]) & set(partitions[right])
    return splits, manifest


def _split_score(candidate, counts, total, fractions, n_rows) -> float:
    """Distance from the target fractions, plus a penalty per class missing from val or test."""
    score = sum(abs(len(candidate[name]) / n_rows - fractions[name]) for name in fractions)
    score += sum(MISSING_HOLDOUT_CLASS_PENALTY for name in ("val", "test") for label in LABELS
                 if not counts[name][label])
    score += sum(abs(counts[name][label] / total[label] - fractions[name]) for name in fractions for label in LABELS)
    return score


def _split_warnings(total, splits) -> list[str]:
    warnings = []
    if min(total.values()) < 20:
        warnings.append("Fewer than 20 examples in at least one class: development results only.")
    for name, part in splits.items():
        absent = sorted(set(LABELS) - {cause_of(row) for row in part})
        if absent:
            warnings.append(f"{name} lacks classes {absent}; those classes cannot be evaluated there.")
    return warnings


# ---------------------------------------------------------------- the experiment

@dataclass
class BankingMLPPredictor:
    """A dev checkpoint with its validation calibration.

    Unlike the comparison models, correctness is calibrated from the *raw* P(cause=none).
    """

    predictor: MLPPredictor
    calibration: dict

    def predict(self, rows):
        cause, correct = self.predictor.predict(rows)
        cause = scale_temperature(cause, self.calibration["cause"]["temperature"])
        binary = scale_temperature(as_binary(correct), self.calibration["correctness"]["temperature"])
        return cause, binary[:, 1]


def load_banking_mlp(path: Path) -> BankingMLPPredictor:
    metadata = read_json(path / MONITOR_FILE)
    if metadata["labels"] != list(LABELS):
        raise ValueError("Expected a banking cause-classification checkpoint")
    return BankingMLPPredictor(load_mlp(path, "cpu"), metadata["calibration"])


def run_banking(run: Path, out: Path, settings: TrainSettings = TrainSettings()) -> dict:
    """Prepare one run, split it, train and calibrate the MLP, and score the test split."""
    require_empty_dir(out, "Use an empty banking experiment directory")
    torch.set_num_threads(settings.threads)
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)

    splits, manifest = _prepare_splits(run, out, settings.seed)
    train, val, test = splits["train"], splits["val"], splits["test"]

    model_dir = out / "mlp"
    details = train_mlp(train, val, model_dir, replace(settings, verbose=True), device="cpu")
    calibration = _calibrate(model_dir, val)

    predictions, metrics = _score_one_at_a_time(model_dir, test)
    _verify_reload(model_dir, test, predictions)
    details["checkpoint_reload_verified"] = True

    majority_probs = majority_probabilities([cause_of(row) for row in train], len(test), LABELS)
    result = {"status": "complete", "task": "banking_monitor", "development_only": True,
              "device": "cpu", "seed": settings.seed, "data": manifest, "training": details,
              "test": metrics, "calibration": calibration,
              "majority_baseline": evaluate(test, majority_probs, majority_probs[:, 0]),
              "elapsed_seconds": time.perf_counter() - started,
              "limitation": LIMITATION}
    write_jsonl(out / "test_predictions.jsonl", predictions)
    write_json(out / "results.json", result)
    return result


def _prepare_splits(run: Path, out: Path, seed: int):
    """Repair and deduplicate the run, split it, and write the splits with their manifest."""
    audit = prepare_training_data(run, out / "prepared")
    rows, equivalent_duplicates = read_rows(out / "prepared" / LAYA_ROWS_FILE)
    splits, manifest = split_rows(rows, seed)

    split_dir = out / "splits"
    split_dir.mkdir()
    for name, part in splits.items():
        write_jsonl(split_dir / f"{name}.jsonl", part)
    manifest.update(audit=audit, equivalent_duplicates_removed=equivalent_duplicates)
    write_json(split_dir / "manifest.json", manifest)
    return splits, manifest


def _calibrate(model_dir: Path, val) -> dict:
    """Fit both temperatures on validation and save them as the checkpoint's monitor metadata."""
    predictor = load_mlp(model_dir)
    cause, _ = predictor.predict(val)
    calibration = fit_calibration(cause, val, "validation only")
    write_json(model_dir / MONITOR_FILE, {"kind": "mlp", "task": "banking_monitor", "labels": list(LABELS),
                                          "calibration": calibration})
    del predictor
    gc.collect()
    return calibration


def _score_one_at_a_time(model_dir: Path, test):
    """Predict each test row on its own (to time it); returns prediction rows and the metrics."""
    monitor = load_banking_mlp(model_dir)
    monitor.predict(test[:1])  # Warm up before timing.

    causes, corrects, latencies, predictions = [], [], [], []
    for row in test:
        started = time.perf_counter()
        cause, correct = monitor.predict([row])
        latency = (time.perf_counter() - started) * 1000

        causes.append(cause[0])
        corrects.append(float(correct[0]))
        latencies.append(latency)
        predictions.append({"input_id": input_id(row), "expected": row["expected"],
                            "cause_probabilities": dict(zip(LABELS, map(float, cause[0]))),
                            "correct_probability": float(correct[0]), "latency_ms": latency})
    return predictions, evaluate(test, causes, corrects, latencies)


def _verify_reload(model_dir: Path, test, predictions) -> None:
    """A freshly loaded checkpoint must reproduce the one-at-a-time predictions."""
    cause, correct = load_banking_mlp(model_dir).predict(test)
    np.testing.assert_allclose(cause, [[p["cause_probabilities"][label] for label in LABELS] for p in predictions],
                               atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(correct, [p["correct_probability"] for p in predictions], atol=1e-6, rtol=1e-6)
