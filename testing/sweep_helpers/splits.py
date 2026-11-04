"""Pool the configured agent runs and write the cross-validation folds every model receives.

For each split seed the request groups are shuffled into k folds, so every group is tested
exactly once per split seed. Inside each fold, a few training groups also go to
``val.jsonl`` for Laya's early stopping; the MLP and stacking add them back to training.
"""
from __future__ import annotations

import random
import re
from collections import Counter, defaultdict

from .config import SplitSettings, Sweep, read_json, write_json, write_jsonl
from mlp_baseline.agent_runs import merge_agent_runs
from mlp_baseline.common import LABELS, input_id, request_group

# How many shuffles to try before giving up on a valid fold assignment.
MAX_ATTEMPTS = 10_000

# request-group key -> the rows of that group
Groups = dict[str, list[dict]]


def make_splits(sweep: Sweep) -> dict:
    """Write every fold's train/val/test files once; later calls reuse them."""
    manifest_path = sweep.out / "splits.json"
    if manifest_path.exists():
        return checked_manifest(sweep, read_json(manifest_path))

    rows = pool_rows(sweep)
    groups = group_rows(rows)
    assignments = sample_assignments(groups, sweep.split, total_rows=len(rows))

    splits = [{"split_seed": split_seed,
               "folds": [write_fold(sweep, groups, split_seed, fold, test_groups)
                         for fold, test_groups in enumerate(folds)]}
              for split_seed, folds in assignments.items()]

    manifest = {
        "config_name": sweep.name,
        "data_fingerprint": sweep.data_fingerprint(),
        "runs": sweep.config["runs"],
        "split_settings": sweep.config["split"],
        "pool": {"rows": len(rows), "groups": len(groups), "labels": label_counts(rows),
                 "group_sizes": {key: len(value) for key, value in sorted(groups.items())}},
        "splits": splits,
    }
    write_json(sweep.out / "config.json", sweep.config)
    write_json(manifest_path, manifest)
    return manifest


def checked_manifest(sweep: Sweep, manifest: dict) -> dict:
    """An existing manifest is only reused if it was built from the same runs and split settings."""
    if manifest["data_fingerprint"] != sweep.data_fingerprint():
        raise ValueError(f"{sweep.out} was built from different runs or split settings; "
                         "give the config a new 'name'")
    return manifest


# ---------------------------------------------------------------- pooling

def pool_rows(sweep: Sweep) -> list[dict]:
    """Merge the configured runs (label repair + dedup) and tag every row with its input_id."""
    merged = merge_agent_runs(sweep.resolve(sweep.config["runs_dir"]), tuple(sweep.config["runs"]))

    rows, seen = [], {}
    for row in merged.rows:
        # Laya's duplicate check is stricter than the MLP's; fail now rather than mid-sweep.
        signature = laya_signature(row)
        if signature in seen:
            raise ValueError(f"Rows {seen[signature]} and {input_id(row)} are duplicates under Laya's check")
        seen[signature] = input_id(row)

        rows.append({"state": row["state"], "questions": row["questions"], "expected": row["expected"],
                     "meta": {"input_id": input_id(row), "source_runs": row["meta"]["source_runs"]}})
    return rows


def laya_signature(row: dict) -> str:
    """Same normalization as ``train_laya.read_splits``."""
    state = re.sub(r"\b(pay|txn)_synth_[0-9a-f]{8}\b", r"\1_synth_<id>", row["state"])
    return " ".join(state.lower().split())


def group_rows(rows: list[dict]) -> Groups:
    groups = defaultdict(list)
    for row in rows:
        groups[request_group(row)].append(row)
    return dict(groups)


# ---------------------------------------------------------------- choosing folds

def sample_assignments(groups: Groups, settings: SplitSettings, *, total_rows: int) -> dict[int, list[list[str]]]:
    """One valid k-fold assignment per split seed, each different from the others."""
    max_test_rows = settings.max_test_fraction * total_rows
    used, assignments = set(), {}
    for split_seed in settings.seeds:
        assignments[split_seed] = sample_folds(groups, settings.k, split_seed, max_test_rows, used)
    return assignments


def sample_folds(groups: Groups, k: int, seed: int, max_test_rows: float, used: set) -> list[list[str]]:
    """Shuffle the groups into k folds until the folds are valid and not used by an earlier seed."""
    rng = random.Random(seed)
    keys = sorted(groups)

    for _ in range(MAX_ATTEMPTS):
        rng.shuffle(keys)
        folds = [sorted(keys[index::k]) for index in range(k)]
        assignment = frozenset(frozenset(fold) for fold in folds)

        if assignment not in used and all(valid_test_fold(fold, groups, max_test_rows) for fold in folds):
            used.add(assignment)
            return folds

    raise ValueError(f"Split seed {seed}: found no new valid {k}-fold assignment")


def valid_test_fold(test_groups: list[str], groups: Groups, max_test_rows: float) -> bool:
    """The test fold and its training part both keep all three causes, and the fold is not too large."""
    test = rows_of(test_groups, groups)
    train = rows_of([key for key in groups if key not in test_groups], groups)
    return len(test) <= max_test_rows and has_every_cause(test) and has_every_cause(train)


def sample_laya_val(train_groups: list[str], groups: Groups, fraction: float, seed: int) -> list[str]:
    """Pick training groups for Laya's validation file; what stays in training keeps every cause."""
    count = max(1, round(len(train_groups) * fraction))
    if count >= len(train_groups):
        raise ValueError("Not enough training groups to hold out Laya validation")

    rng = random.Random(seed)
    keys = list(train_groups)
    for _ in range(MAX_ATTEMPTS):
        rng.shuffle(keys)
        if has_every_cause(rows_of(keys[count:], groups)):
            return sorted(keys[:count])

    raise ValueError("No Laya validation choice leaves every cause in training")


# ---------------------------------------------------------------- writing one fold

def write_fold(sweep: Sweep, groups: Groups, split_seed: int, fold: int, test_groups: list[str]) -> dict:
    """Write one fold's train/val/test files and return its manifest entry."""
    train_groups = [key for key in sorted(groups) if key not in test_groups]
    # Seeded per fold, so the validation choice does not depend on the training seed.
    val_groups = sample_laya_val(train_groups, groups, sweep.split.laya_val_fraction,
                                 seed=split_seed * 1000 + fold)

    parts = {
        "train": rows_of([key for key in train_groups if key not in val_groups], groups),
        "val": rows_of(val_groups, groups),
        "test": rows_of(test_groups, groups),
    }

    files = sweep.fold_files(split_seed, fold)
    files.dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(files.train, parts["train"])
    write_jsonl(files.val, parts["val"])
    write_jsonl(files.test, parts["test"])

    return {
        "fold": fold,
        "test_groups": test_groups,
        "laya_val_groups": val_groups,
        **{f"{name}_rows": len(part) for name, part in parts.items()},
        **{f"{name}_labels": label_counts(part) for name, part in parts.items()},
    }


# ---------------------------------------------------------------- small helpers

def rows_of(keys, groups: Groups) -> list[dict]:
    return [row for key in keys for row in groups[key]]


def has_every_cause(rows) -> bool:
    return {row["expected"]["cause"] for row in rows} == set(LABELS)


def label_counts(rows) -> dict:
    return dict(sorted(Counter(row["expected"]["cause"] for row in rows).items()))
