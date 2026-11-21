"""Pool the configured agent runs and write the cross-validation folds every model receives.

For each split seed the request groups are shuffled into k folds, so every group is tested
exactly once per split seed. Inside each fold, a few training groups also go to
``val.jsonl`` for Laya's early stopping; the MLP and stacking add them back to training.

Two optional config keys narrow this (without them the sweep is unchanged):

- ``exclude_tools``: drop the rows of these tools from the pool, e.g. ``["initiate_transfer"]`` when a
  deployment's monitors are never asked about transfers.
- ``test_runs``: only rows from these runs are ever tested. Only the request groups they contain are
  shuffled into folds, and a fold's test file holds just those runs' rows of its groups. Every other row
  is training data; rows of other runs that share a request group with a test fold are left out of that
  fold, so no group is on both sides.
- ``test_exclude_tools`` (with ``test_runs``): never test these tools' rows; they can still train.
- ``train_on_test_runs`` (with ``test_runs``, default true): false keeps every row of the test runs out
  of training, to measure a model trained without any of that data.
- ``exclude_hardening_blocked``: drop calls that MCP hardening rejected (``meta.hardening.failure_reason``
  in the run's ``examples.jsonl``). In a run with recovery the monitors are never asked about those: the
  call already failed, with an empty output. Runs made before hardening have no such calls.
- ``test_required_causes`` (default: all of them): the causes every test fold must contain. Training parts
  still need every cause. Lower it when a cause has almost vanished from the test data.
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

# Optional config keys that change which rows are pooled, tested or trained on.
FILTER_KEYS = ("exclude_tools", "test_runs", "test_exclude_tools", "train_on_test_runs", "exclude_hardening_blocked",
               "test_required_causes")

# request-group key -> the rows of that group
Groups = dict[str, list[dict]]


def make_splits(sweep: Sweep) -> dict:
    """Write every fold's train/val/test files once; later calls reuse them."""
    manifest_path = sweep.out / "splits.json"
    if manifest_path.exists():
        return checked_manifest(sweep, read_json(manifest_path))

    rows = pool_rows(sweep)
    groups = group_rows(rows)
    test_runs = set(sweep.config.get("test_runs") or ())
    if test_runs:
        unknown = test_runs - set(sweep.config["runs"])
        if unknown:
            raise ValueError(f"test_runs {sorted(unknown)} are not in runs")
        untested = set(sweep.config.get("test_exclude_tools") or ())
        testable = {key: kept for key, members in groups.items()
                    if (kept := [row for row in members if is_from(row, test_runs) and row_tool(row) not in untested])}
        if not sweep.config.get("train_on_test_runs", True):
            groups = {key: kept for key, members in groups.items()
                      if (kept := [row for row in members if not is_from(row, test_runs)])}
        assignments = sample_assignments(testable, sweep.split, total_rows=sum(map(len, testable.values())),
                                         train_groups=groups,
                                         test_causes=set(sweep.config.get("test_required_causes") or LABELS))
    else:
        testable = groups
        assignments = sample_assignments(groups, sweep.split, total_rows=len(rows))

    splits = [{"split_seed": split_seed,
               "folds": [write_fold(sweep, groups, split_seed, fold, test_groups, test_rows=testable)
                         for fold, test_groups in enumerate(folds)]}
              for split_seed, folds in assignments.items()]

    manifest = {
        "config_name": sweep.name,
        "data_fingerprint": sweep.data_fingerprint(),
        "runs": sweep.config["runs"],
        "split_settings": sweep.config["split"],
        **{key: sweep.config[key] for key in FILTER_KEYS if key in sweep.config},
        "pool": {"rows": len(rows), "groups": len(groups), "labels": label_counts(rows),
                 "group_sizes": {key: len(value) for key, value in sorted(groups.items())}},
        "splits": splits,
    }
    write_json(sweep.out / "config.json", sweep.config)
    write_json(manifest_path, manifest)
    return manifest


def checked_manifest(sweep: Sweep, manifest: dict) -> dict:
    """An existing manifest is only reused if it was built from the same runs and split settings."""
    if manifest["data_fingerprint"] == sweep.data_fingerprint():
        return manifest
    # Older results hashed runs_dir too; check them against the runs_dir they were built with.
    saved = sweep.out / "config.json"
    saved_runs_dir = read_json(saved).get("runs_dir") if saved.exists() else None
    if saved_runs_dir is not None and manifest["data_fingerprint"] == sweep.legacy_data_fingerprint(saved_runs_dir):
        return manifest
    raise ValueError(f"{sweep.out} was built from different runs or split settings; "
                     "give the config a new 'name'")


# ---------------------------------------------------------------- pooling

def pool_rows(sweep: Sweep) -> list[dict]:
    """Merge the configured runs (label repair + dedup) and tag every row with its input_id."""
    merged = merge_agent_runs(sweep.resolve(sweep.config["runs_dir"]), tuple(sweep.config["runs"]))

    excluded = set(sweep.config.get("exclude_tools") or ())
    blocked = hardening_blocked(sweep, merged) if sweep.config.get("exclude_hardening_blocked") else set()
    rows, seen = [], {}
    for row in merged.rows:
        if excluded and row_tool(row) in excluded:
            continue
        if input_id(row) in blocked:
            continue
        # Laya's duplicate check is stricter than the MLP's; fail now rather than mid-sweep.
        signature = laya_signature(row)
        if signature in seen:
            raise ValueError(f"Rows {seen[signature]} and {input_id(row)} are duplicates under Laya's check")
        seen[signature] = input_id(row)

        rows.append({"state": row["state"], "questions": row["questions"], "expected": row["expected"],
                     "meta": {"input_id": input_id(row), "source_runs": row["meta"]["source_runs"]}})
    return rows


def hardening_blocked(sweep: Sweep, merged) -> set[str]:
    """input_ids of rows whose call hardening rejected in any run they came from."""
    from .config import read_jsonl

    runs_dir = sweep.resolve(sweep.config["runs_dir"])
    failed: dict[str, set[int]] = {}
    for run in sweep.config["runs"]:
        examples = read_jsonl(runs_dir / run / "examples.jsonl")
        failed[run] = {line for line, example in enumerate(examples, 1)
                       if (example.get("meta", {}).get("hardening") or {}).get("failure_reason")}
    return {key for key, sources in merged.provenance.items()
            if any(source["source_line"] in failed[source["run"]] for source in sources)}


def row_tool(row: dict) -> str | None:
    """The tool a monitor row is about (its state's ``tool:`` line)."""
    return next((line[len("tool: "):].strip() for line in row["state"].splitlines() if line.startswith("tool: ")), None)


def is_from(row: dict, runs: set) -> bool:
    return bool(set(row["meta"]["source_runs"]) & runs)


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

def sample_assignments(groups: Groups, settings: SplitSettings, *, total_rows: int,
                       train_groups: Groups | None = None, test_causes: set | None = None) -> dict[int, list[list[str]]]:
    """One valid k-fold assignment per split seed, each different from the others.

    ``groups`` are the rows that can be tested; ``train_groups`` (default: the same) are all the rows,
    so a fold's training part is checked as it will really be written.
    """
    max_test_rows = settings.max_test_fraction * total_rows
    used, assignments = set(), {}
    for split_seed in settings.seeds:
        assignments[split_seed] = sample_folds(groups, settings.k, split_seed, max_test_rows, used,
                                               train_groups=train_groups, test_causes=test_causes)
    return assignments


def sample_folds(groups: Groups, k: int, seed: int, max_test_rows: float, used: set,
                 train_groups: Groups | None = None, test_causes: set | None = None) -> list[list[str]]:
    """Shuffle the groups into k folds until the folds are valid and not used by an earlier seed."""
    rng = random.Random(seed)
    keys = sorted(groups)

    for _ in range(MAX_ATTEMPTS):
        rng.shuffle(keys)
        folds = [sorted(keys[index::k]) for index in range(k)]
        assignment = frozenset(frozenset(fold) for fold in folds)

        if assignment not in used and all(valid_test_fold(fold, groups, max_test_rows, train_groups, test_causes)
                                          for fold in folds):
            used.add(assignment)
            return folds

    raise ValueError(f"Split seed {seed}: found no new valid {k}-fold assignment")


def valid_test_fold(test_groups: list[str], groups: Groups, max_test_rows: float,
                    train_groups: Groups | None = None, test_causes: set | None = None) -> bool:
    """The test fold has ``test_causes`` (default: all three), its training part has all three, and the
    fold is not too large."""
    train_groups = train_groups if train_groups is not None else groups
    test = rows_of(test_groups, groups)
    train = rows_of([key for key in train_groups if key not in test_groups], train_groups)
    has_test_causes = (has_every_cause(test) if test_causes is None
                       else test_causes <= {row["expected"]["cause"] for row in test})
    return len(test) <= max_test_rows and has_test_causes and has_every_cause(train)


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

def write_fold(sweep: Sweep, groups: Groups, split_seed: int, fold: int, test_groups: list[str],
               test_rows: Groups | None = None) -> dict:  # groups: the rows that may train
    """Write one fold's train/val/test files and return its manifest entry.

    ``test_rows`` (default: ``groups``) holds the rows that may be tested; with ``test_runs`` it has only
    those runs' rows, and the other rows of a test group go nowhere in this fold.
    """
    test_rows = test_rows if test_rows is not None else groups
    train_groups = [key for key in sorted(groups) if key not in test_groups]
    # Seeded per fold, so the validation choice does not depend on the training seed.
    val_groups = sample_laya_val(train_groups, groups, sweep.split.laya_val_fraction,
                                 seed=split_seed * 1000 + fold)

    parts = {
        "train": rows_of([key for key in train_groups if key not in val_groups], groups),
        "val": rows_of(val_groups, groups),
        "test": rows_of(test_groups, test_rows),
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
