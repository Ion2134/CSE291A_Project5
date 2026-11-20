"""Banking-monitor datasets built from merged agent runs.

- ``combine_agent_runs``: merged runs -> one group-disjoint train/test split (``prepare-agent-data``)
- ``validate_combined_dataset``: reload such a dataset and check it
- ``export_laya_banking``: the same split in Laya's train/val/test format (``export-laya-banking``)
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from .agent_runs import COMBINED_RUNS, merge_agent_runs
from .common import (LABELS, cause_counts, cause_of, check_monitor_row, input_id, read_json, read_jsonl,
                     request_group, require_empty_dir, sha256_file, write_json, write_jsonl)
from .splits import stratified_group_partition

TASK = "banking_monitor"
COMBINED_SPLITS = ("train", "test")
# Below this many rows in some class, results are flagged as development-only.
MIN_CLASS_ROWS = 20
SMALL_CLASS_WARNING = "Fewer than 20 unique examples in at least one class: development results only."
EVAL_POLICY = "First-party agent runs only; external datasets never train or evaluate on this artifact."


# ---------------------------------------------------------------- combined train/test dataset

def combine_agent_runs(runs_dir: Path, out_dir: Path, *, seed: int = 42, test_fraction: float = 0.2,
                       runs=COMBINED_RUNS) -> dict:
    """Merge ``runs`` and write a group-disjoint train/test split, provenance and a manifest."""
    runs_dir, out_dir = Path(runs_dir), Path(out_dir)
    require_empty_dir(out_dir, "Use a new or empty combined-dataset directory")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in (0, 1)")

    merged = merge_agent_runs(runs_dir, runs)
    rows = merged.rows
    label_counts = Counter(cause_of(row) for row in rows)

    groups = sorted({request_group(row) for row in rows})
    n_test = max(1, round(len(groups) * test_fraction))
    if n_test >= len(groups):
        raise ValueError("Not enough independent request groups to hold out a test part")
    splits, assignment = stratified_group_partition(rows, {"train": len(groups) - n_test, "test": n_test},
                                                    seed=seed)

    manifest = {
        "task": TASK,
        "seed": seed,
        "strategy": "repair_per_run_then_merge_dedup_then_group_stratified_split",
        "eval_policy": EVAL_POLICY,
        "runs": merged.run_reports,
        "raw_rows": merged.raw_rows,
        "repaired_rows": len(merged.corrections),
        "label_corrections": merged.corrections,
        "unique_inputs": len(rows),
        "duplicate_rows_removed": merged.raw_rows - len(rows),
        "label_counts": dict(sorted(label_counts.items())),
        "groups": len(groups),
        "test_fraction_requested": test_fraction,
        "splits": {name: _split_entry(splits[name], assignment[name]) for name in splits},
        "warnings": [SMALL_CLASS_WARNING] if min(label_counts.values()) < MIN_CLASS_ROWS else [],
    }
    left, right = set(manifest["splits"]["train"]["input_ids"]), set(manifest["splits"]["test"]["input_ids"])
    assert not left & right and not set(assignment["train"]) & set(assignment["test"])

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in splits:
        write_jsonl(out_dir / f"{name}.jsonl", splits[name])
    write_jsonl(out_dir / "provenance.jsonl",
                [{"input_id": key, "sources": merged.provenance[key]} for key in sorted(merged.provenance)])
    manifest["split_sha256"] = {name: sha256_file(out_dir / f"{name}.jsonl") for name in splits}
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def validate_combined_dataset(data: Path):
    """Reload a combined dataset and verify hashes, schema, classes and split isolation."""
    data = Path(data)
    manifest = read_json(data / "manifest.json")
    if manifest["task"] != TASK:
        raise ValueError("Expected a combined first-party monitor dataset")

    splits = {name: read_jsonl(data / f"{name}.jsonl") for name in COMBINED_SPLITS}
    for name, rows in splits.items():
        if not rows:
            raise ValueError(f"{name} split is empty")
        if sha256_file(data / f"{name}.jsonl") != manifest["split_sha256"][name]:
            raise ValueError(f"{name} data hash differs from its manifest")
        questions = rows[0]["questions"]
        for index, row in enumerate(rows, 1):
            check_monitor_row(row, name, index, questions)
        if {cause_of(row) for row in rows} != set(LABELS):
            raise ValueError(f"{name} must contain every cause class")

    if {input_id(row) for row in splits["train"]} & {input_id(row) for row in splits["test"]}:
        raise ValueError("Identical input leaked across the combined train/test split")
    if {request_group(row) for row in splits["train"]} & {request_group(row) for row in splits["test"]}:
        raise ValueError("User-request group leaked across the combined train/test split")
    return splits, manifest


# ---------------------------------------------------------------- Laya export

def export_laya_banking(combined: Path, out_dir: Path, *, seed: int = 42, val_fraction: float = 0.2) -> dict:
    """Export a combined dataset as Laya train/val/test JSONL.

    The test split is the combined dataset's own, so the MLP and Laya are scored on the same rows;
    validation is carved from its train pool with the same group-stratified partitioner.
    """
    combined, out_dir = Path(combined), Path(out_dir)
    require_empty_dir(out_dir, "Use a new or empty Laya banking export directory")
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0, 1)")

    source_splits, source = validate_combined_dataset(combined)
    train_pool, test = source_splits["train"], source_splits["test"]
    groups = sorted({request_group(row) for row in train_pool})
    n_val = max(1, round(len(groups) * val_fraction))
    if n_val >= len(groups):
        raise ValueError("Not enough train-pool request groups to hold out a validation part")
    parts, assignment = stratified_group_partition(train_pool, {"train": len(groups) - n_val, "val": n_val},
                                                   seed=seed)
    splits = {"train": parts["train"], "val": parts["val"], "test": test}
    assignment = {**assignment, "test": source["splits"]["test"]["groups"]}

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        write_jsonl(out_dir / f"{name}.jsonl", [_laya_row(row) for row in rows])

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
        "splits": {name: _split_entry(splits[name], assignment[name]) for name in splits},
        "warnings": _export_warnings(source, splits["train"]),
    }
    ids = {name: set(manifest["splits"][name]["input_ids"]) for name in splits}
    assert not (ids["train"] & ids["val"]) and not (ids["train"] & ids["test"]) and not (ids["val"] & ids["test"])
    assert ids["test"] == set(source["splits"]["test"]["input_ids"])

    manifest["split_sha256"] = {name: sha256_file(out_dir / f"{name}.jsonl") for name in splits}
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def _laya_row(row: dict) -> dict:
    """Laya's banking rows keep only the monitor fields (no provenance ``meta``)."""
    return {"state": row["state"], "questions": row["questions"], "expected": row["expected"]}


def _export_warnings(source: dict, train: list[dict]) -> list[str]:
    warnings = list(source.get("warnings") or [])
    warnings.append("Validation carved from the MLP train pool; test rows match the source combine artifact.")
    if min(Counter(cause_of(row) for row in train).values()) < MIN_CLASS_ROWS:
        warnings.append("Fewer than 20 unique train examples in at least one class: development results only.")
    return warnings


def _split_entry(rows: list[dict], groups: list[str]) -> dict:
    """Manifest entry for one split: its size, request groups, row ids and cause counts."""
    return {"rows": len(rows), "groups": groups, "input_ids": [input_id(row) for row in rows],
            "label_counts": cause_counts(rows)}
