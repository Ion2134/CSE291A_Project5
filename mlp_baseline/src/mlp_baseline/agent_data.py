"""Combine first-party agent runs into a single pruned monitor dataset.

Every run contributes raw ``laya_train.jsonl`` rows. Each run is repaired with
the legacy no-op IO fix, merged across runs with provenance, deduplicated by
normalized input identity, and split once into group-disjoint train/test
parts. External datasets never enter this pipeline; evaluation is first-party
only.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from .banking_data import _read_jsonl, _repair_noop_io_label
from .cv import stratified_group_partition
from .data import LABELS, input_id, request_group, write_json, write_jsonl

COMBINED_RUNS = (
    "agent-qwen36a3b",
    "agent42",
    "agent42-v2",
    "triton42",
    "triton42-v2",
    "triton44",
    "triton45",
    "triton46",
    "triton47",
    "triton48",
)
TASK = "banking_monitor"


def _check_row(row: dict, run_name: str, line: int, questions) -> None:
    expected = row["expected"]
    if (set(expected) != {"output_correct", "cause"}
            or type(expected["output_correct"]) is not bool
            or expected["cause"] not in LABELS
            or expected["output_correct"] != (expected["cause"] == "none")):
        raise ValueError(f"Invalid or inconsistent monitor labels at {run_name} row {line}")
    if questions is not None and row["questions"] != questions:
        raise ValueError("All runs must use the same monitor question schema")


def _load_run(run_dir: Path, run_name: str, questions):
    source = run_dir / "laya_train.jsonl"
    rows = _read_jsonl(source)
    if not rows:
        raise ValueError(f"{run_name}: no training rows found")
    examples_path = run_dir / "examples.jsonl"
    examples = _read_jsonl(examples_path)
    if len(examples) != len(rows):
        raise ValueError(f"{run_name}: examples.jsonl and laya_train.jsonl have different row counts")
    repaired, corrections = [], []
    for line, (row, example) in enumerate(zip(rows, examples), 1):
        expected = {key: example["label"][key] for key in ("output_correct", "cause")}
        if example["input"]["state_text"] != row["state"] or expected != row["expected"]:
            raise ValueError(f"{run_name}: source files disagree at row {line}")
        _check_row(row, run_name, line, questions)
        fixed = _repair_noop_io_label(row, example)
        if fixed != row:
            corrections.append({
                "run": run_name, "source_line": line, "example_id": example["example_id"],
                "before": row["expected"], "after": fixed["expected"],
                "reason": "IO-mismatch attack did not change a successful on-mandate call",
            })
        repaired.append((fixed, {"run": run_name, "source_line": line, "example_id": example["example_id"]}))
    hashes = {"laya_train_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "examples_sha256": hashlib.sha256(examples_path.read_bytes()).hexdigest()}
    return repaired, corrections, hashes


def read_runs_config(path: Path) -> dict:
    """Read a shared JSON config (e.g. ``testing/configs/*.json``) naming the agent runs to pool.

    Only ``runs`` is required here; other keys belong to the sweep runner.
    """
    path = Path(path)
    config = json.loads(path.read_text())
    if not config.get("runs") or not all(isinstance(name, str) for name in config["runs"]):
        raise ValueError(f"{path}: 'runs' must be a non-empty list of run directory names")
    return config


def merge_agent_runs(runs_dir: Path, runs=COMBINED_RUNS) -> dict:
    """Repair each run, then merge identical normalized inputs across runs.

    Returns the unique rows (each with ``meta.source_runs``) plus the
    provenance and per-run reports. Nothing is split or written here.
    """
    runs_dir = Path(runs_dir)
    questions = None
    entries, corrections, run_reports = [], [], {}
    for name in runs:
        repaired, run_corrections, hashes = _load_run(runs_dir / name, name, questions)
        if questions is None and repaired:
            questions = repaired[0][0]["questions"]
        corrections.extend(run_corrections)
        run_reports[name] = {"raw_rows": len(repaired), "repaired_rows": len(run_corrections), **hashes}
        baseline_path = runs_dir / name / "baseline.json"
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text())
            if baseline.get("model"):
                run_reports[name]["model"] = baseline["model"]
        entries.extend(repaired)

    # Merge identical normalized inputs; conflicting answers remain a hard error.
    unique: dict[str, dict] = {}
    provenance: dict[str, list] = {}
    for row, source in entries:
        key = input_id(row)
        if key in unique:
            if unique[key]["expected"] != row["expected"]:
                first = provenance[key][0]
                raise ValueError(
                    f"Unresolved conflicting labels for equivalent inputs at "
                    f"{first['run']} row {first['source_line']} and {source['run']} row {source['source_line']}")
            unique[key]["meta"]["source_runs"] = sorted(set(unique[key]["meta"]["source_runs"]) | {source["run"]})
            provenance[key].append(source)
        else:
            merged = {**row, "meta": {"source_runs": [source["run"]]}}
            unique[key] = merged
            provenance[key] = [source]
    rows = list(unique.values())
    if {row["expected"]["cause"] for row in rows} != set(LABELS):
        raise ValueError("The combined dataset must include correct, agent-error, and adversary-error examples")
    return {"rows": rows, "provenance": provenance, "run_reports": run_reports,
            "corrections": corrections, "raw_rows": len(entries)}


def combine_agent_runs(runs_dir: Path, out_dir: Path, *, seed: int = 42, test_fraction: float = 0.2,
                       runs=COMBINED_RUNS) -> dict:
    """Write train/test JSONL + provenance + manifest; raw runs stay untouched."""
    runs_dir, out_dir = Path(runs_dir), Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("Use a new or empty combined-dataset directory")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in (0, 1)")

    merged = merge_agent_runs(runs_dir, runs)
    rows, provenance = merged["rows"], merged["provenance"]
    corrections, raw_rows = merged["corrections"], merged["raw_rows"]
    label_counts = Counter(row["expected"]["cause"] for row in rows)

    groups = sorted({request_group(row) for row in rows})
    n_test = max(1, round(len(groups) * test_fraction))
    if n_test >= len(groups):
        raise ValueError("Not enough independent request groups to hold out a test part")
    sizes = {"train": len(groups) - n_test, "test": n_test}
    splits, assignment = stratified_group_partition(rows, sizes, seed=seed)

    warnings = []
    if min(label_counts.values()) < 20:
        warnings.append("Fewer than 20 unique examples in at least one class: development results only.")
    manifest = {
        "task": TASK,
        "seed": seed,
        "strategy": "repair_per_run_then_merge_dedup_then_group_stratified_split",
        "eval_policy": "First-party agent runs only; external datasets never train or evaluate on this artifact.",
        "runs": merged["run_reports"],
        "raw_rows": raw_rows,
        "repaired_rows": len(corrections),
        "label_corrections": corrections,
        "unique_inputs": len(rows),
        "duplicate_rows_removed": raw_rows - len(rows),
        "label_counts": dict(sorted(label_counts.items())),
        "groups": len(groups),
        "test_fraction_requested": test_fraction,
        "splits": {name: {
            "rows": len(splits[name]),
            "groups": assignment[name],
            "input_ids": [input_id(row) for row in splits[name]],
            "label_counts": dict(sorted(Counter(row["expected"]["cause"] for row in splits[name]).items())),
        } for name in splits},
        "warnings": warnings,
    }
    left, right = set(manifest["splits"]["train"]["input_ids"]), set(manifest["splits"]["test"]["input_ids"])
    assert not left & right and not set(assignment["train"]) & set(assignment["test"])

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in splits:
        write_jsonl(out_dir / f"{name}.jsonl", splits[name])
    write_jsonl(out_dir / "provenance.jsonl",
                [{"input_id": key, "sources": provenance[key]} for key in sorted(provenance)])
    manifest["split_sha256"] = {name: hashlib.sha256((out_dir / f"{name}.jsonl").read_bytes()).hexdigest()
                                for name in splits}
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def validate_combined_dataset(data: Path):
    """Reload a combined dataset and verify hashes, schema, and split isolation."""
    data = Path(data)
    manifest = json.loads((data / "manifest.json").read_text())
    if manifest["task"] != TASK:
        raise ValueError("Expected a combined first-party monitor dataset")
    splits = {name: _read_jsonl(data / f"{name}.jsonl") for name in ("train", "test")}
    for name, rows in splits.items():
        if not rows:
            raise ValueError(f"{name} split is empty")
        if hashlib.sha256((data / f"{name}.jsonl").read_bytes()).hexdigest() != manifest["split_sha256"][name]:
            raise ValueError(f"{name} data hash differs from its manifest")
        questions = rows[0]["questions"]
        for index, row in enumerate(rows, 1):
            _check_row(row, name, index, questions)
        if {row["expected"]["cause"] for row in rows} != set(LABELS):
            raise ValueError(f"{name} must contain every cause class")
    if {input_id(row) for row in splits["train"]} & {input_id(row) for row in splits["test"]}:
        raise ValueError("Identical input leaked across the combined train/test split")
    if {request_group(row) for row in splits["train"]} & {request_group(row) for row in splits["test"]}:
        raise ValueError("User-request group leaked across the combined train/test split")
    return splits, manifest


def _laya_row(row: dict) -> dict:
    """Laya banking JSONL keeps only the monitor fields (no provenance meta)."""
    return {"state": row["state"], "questions": row["questions"], "expected": row["expected"]}


def export_laya_banking(combined: Path, out_dir: Path, *, seed: int = 42,
                        val_fraction: float = 0.2) -> dict:
    """Export a combined agent dataset as Laya train/val/test JSONL.

    The held-out test split is kept identical to the MLP combine artifact so
    both models evaluate on the same rows. Validation is carved from the MLP
    train pool with the same group-stratified partitioner.
    """
    combined, out_dir = Path(combined), Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("Use a new or empty Laya banking export directory")
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0, 1)")

    source_splits, source = validate_combined_dataset(combined)
    train_pool, test = source_splits["train"], source_splits["test"]
    groups = sorted({request_group(row) for row in train_pool})
    n_val = max(1, round(len(groups) * val_fraction))
    if n_val >= len(groups):
        raise ValueError("Not enough train-pool request groups to hold out a validation part")
    sizes = {"train": len(groups) - n_val, "val": n_val}
    parts, assignment = stratified_group_partition(train_pool, sizes, seed=seed)
    splits = {"train": parts["train"], "val": parts["val"], "test": test}

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        write_jsonl(out_dir / f"{name}.jsonl", [_laya_row(row) for row in rows])

    warnings = list(source.get("warnings") or [])
    warnings.append(
        "Validation carved from the MLP train pool; test rows match the source combine artifact.")
    if min(Counter(row["expected"]["cause"] for row in splits["train"]).values()) < 20:
        warnings.append("Fewer than 20 unique train examples in at least one class: development results only.")

    manifest = {
        "task": "banking",
        "format": "laya",
        "source_combined": str(combined.resolve()),
        "source_task": source["task"],
        "source_seed": source["seed"],
        "source_runs": source["runs"],
        "source_unique_inputs": source["unique_inputs"],
        "source_split_sha256": source["split_sha256"],
        "seed": seed,
        "val_fraction_requested": val_fraction,
        "strategy": "freeze_combined_test_carve_val_from_train_group_stratified",
        "eval_policy": source["eval_policy"],
        "splits": {name: {
            "rows": len(splits[name]),
            "groups": (assignment[name] if name in assignment
                       else source["splits"]["test"]["groups"]),
            "input_ids": [input_id(row) for row in splits[name]],
            "label_counts": dict(sorted(Counter(row["expected"]["cause"] for row in splits[name]).items())),
        } for name in splits},
        "warnings": warnings,
    }
    train_ids = set(manifest["splits"]["train"]["input_ids"])
    val_ids = set(manifest["splits"]["val"]["input_ids"])
    test_ids = set(manifest["splits"]["test"]["input_ids"])
    assert not (train_ids & val_ids) and not (train_ids & test_ids) and not (val_ids & test_ids)
    assert set(manifest["splits"]["test"]["input_ids"]) == set(source["splits"]["test"]["input_ids"])
    manifest["split_sha256"] = {name: hashlib.sha256((out_dir / f"{name}.jsonl").read_bytes()).hexdigest()
                                for name in splits}
    write_json(out_dir / "manifest.json", manifest)
    return manifest
