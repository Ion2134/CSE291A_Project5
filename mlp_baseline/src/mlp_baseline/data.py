from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

LABELS = ("none", "agent", "adversary")


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))


def input_id(row: dict) -> str:
    # These identifiers are random bookkeeping values, not independent tasks.
    state = re.sub(r"\b(pay|txn)_synth_[0-9a-f]{8}\b", r"\1_synth_<id>", row["state"])
    text = json.dumps({"state": state, "questions": row["questions"]}, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def request_group(row: dict) -> str:
    request = next((line[14:] for line in row["state"].splitlines() if line.startswith("user_message: ")), None)
    if request is None:
        raise ValueError("Each monitor state must contain a user_message line for grouped splitting")
    request = re.sub(r"\b(acct|payee)_synth_\w+\b", r"\1_synth_<id>", request.lower())
    request = re.sub(r"\d+(?:\.\d+)?", "<number>", request)
    return hashlib.sha256(request.encode()).hexdigest()


def read_rows(path: Path) -> tuple[list[dict], int]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("Training data is empty")
    seen = {}
    for index, row in enumerate(rows, 1):
        expected = row["expected"]
        if (set(expected) != {"output_correct", "cause"}
                or type(expected["output_correct"]) is not bool
                or expected["cause"] not in LABELS
                or expected["output_correct"] != (expected["cause"] == "none")):
            raise ValueError(f"Invalid or inconsistent monitor labels at row {index}")
        if row["questions"] != rows[0]["questions"]:
            raise ValueError("All rows must use the same monitor question schema")
        if set(row["questions"]) != {"output_correct", "cause"}:
            raise ValueError("Expected output_correct and cause questions")
        key = input_id(row)
        if key in seen and seen[key]["expected"] != expected:
            raise ValueError(f"Conflicting labels for equivalent inputs at row {index}")
        seen.setdefault(key, row)
    return list(seen.values()), len(rows) - len(seen)


def split_rows(rows: list[dict], seed: int = 42, val_fraction: float = 0.2,
               test_fraction: float = 0.2) -> tuple[dict[str, list[dict]], dict]:
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
    total = Counter(row["expected"]["cause"] for row in rows)
    if set(total) != set(LABELS):
        raise ValueError("The dataset must include correct, agent-error, and adversary-error examples")
    rng = random.Random(seed)
    best = None
    # Stratify whole request groups. Never split calls from a request across sets.
    for _ in range(1000):
        order = keys.copy()
        rng.shuffle(order)
        partitions = {"val": order[:n_val], "test": order[n_val:n_val + n_test], "train": order[n_val + n_test:]}
        candidate = {name: [row for key in selected for row in groups[key]] for name, selected in partitions.items()}
        counts = {name: Counter(row["expected"]["cause"] for row in part) for name, part in candidate.items()}
        if set(counts["train"]) != set(LABELS):
            continue
        fractions = {"train": 1 - val_fraction - test_fraction, "val": val_fraction, "test": test_fraction}
        score = sum(abs(len(candidate[name]) / len(rows) - fractions[name]) for name in fractions)
        score += sum(10 for name in ("val", "test") for label in LABELS if not counts[name][label])
        score += sum(abs(counts[name][label] / total[label] - fractions[name])
                     for name in fractions for label in LABELS)
        if best is None or score < best[0]:
            best = (score, candidate, partitions)
    if best is None:
        raise ValueError("Cannot keep all classes in training with independent holdouts; generate more request groups")
    _, splits, partitions = best
    warnings = []
    if min(total.values()) < 20:
        warnings.append("Fewer than 20 examples in at least one class: development results only.")
    for name, part in splits.items():
        absent = sorted(set(LABELS) - {row["expected"]["cause"] for row in part})
        if absent:
            warnings.append(f"{name} lacks classes {absent}; those classes cannot be evaluated there.")
    manifest = {
        "seed": seed, "strategy": "group_by_normalized_user_request",
        "fractions_requested": {"val": val_fraction, "test": test_fraction},
        "splits": {name: {
            "rows": len(part), "groups": partitions[name],
            "input_ids": [input_id(row) for row in part],
            "label_counts": dict(Counter(row["expected"]["cause"] for row in part)),
        } for name, part in splits.items()},
        "warnings": warnings,
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not set(manifest["splits"][left]["input_ids"]) & set(manifest["splits"][right]["input_ids"])
        assert not set(partitions[left]) & set(partitions[right])
    return splits, manifest
