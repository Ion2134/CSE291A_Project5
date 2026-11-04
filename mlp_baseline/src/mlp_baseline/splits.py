"""Group-disjoint stratified partitions and cross-validation folds.

Splits never separate rows that share a normalized user request (``request_group``), and every
part keeps every cause class whenever the data allows it.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict

from .common import LABELS, cause_of, request_group

# Random shuffles tried per partition; the best-stratified valid one wins.
PARTITION_ATTEMPTS = 4000


def stratified_group_partition(rows: list[dict], sizes: dict[str, int], *, seed: int = 42,
                               attempts: int = PARTITION_ATTEMPTS, labels=LABELS):
    """Partition rows into named group-disjoint parts with stratified labels.

    ``sizes`` maps part names to the number of request groups each receives. Returns
    ``(parts, assignment)``: part name -> rows, and part name -> request-group keys. Every part
    must contain every label; raises ValueError when no shuffle within ``attempts`` achieves that.
    """
    groups = _by_group(rows)
    keys = sorted(groups)
    if len(sizes) < 2 or min(sizes.values()) < 1 or sum(sizes.values()) != len(keys):
        raise ValueError("Part sizes must be positive and account for every request group")
    total = Counter(cause_of(row) for row in rows)
    if set(total) != set(labels):
        raise ValueError("The dataset must include every cause class")

    fractions = {name: count / len(keys) for name, count in sizes.items()}
    rng = random.Random(seed)
    best = None
    for _ in range(attempts):
        order = keys.copy()
        rng.shuffle(order)
        assignment = _cut(order, sizes)
        parts = {name: [row for key in selected for row in groups[key]] for name, selected in assignment.items()}
        counts = {name: Counter(cause_of(row) for row in part) for name, part in parts.items()}

        # Hard constraint: no part may lose a class. Among valid ones, the best-stratified wins.
        if any(set(counts[name]) != set(labels) for name in parts):
            continue
        score = _stratification_error(parts, counts, total, fractions, len(rows), labels)
        if best is None or score < best[0]:
            best = (score, parts, assignment)

    if best is None:
        raise ValueError("No group-disjoint partition keeps every class in every part")
    _, parts, assignment = best
    return parts, assignment


def group_kfold(rows: list[dict], k: int = 5, *, seed: int = 42, labels=LABELS) -> dict:
    """Build k group-disjoint stratified cross-validation folds.

    Starts from the requested ``k`` and steps down toward 2 until every fold can hold every
    class; the ``k`` used and the reason for any reduction are in the result.
    """
    groups = _by_group(rows)
    if len(groups) < 2:
        raise ValueError("Need at least two request groups for cross-validation")
    if k < 2:
        raise ValueError("k must be at least 2")

    k_used, assignment = _largest_valid_kfold(rows, len(groups), k, seed, labels)
    folds = []
    for name in sorted(assignment):
        held_out = set(assignment[name])
        folds.append({
            "name": name,
            "train": [row for key, group in groups.items() if key not in held_out for row in group],
            "val": [row for key in assignment[name] for row in groups[key]],
            "val_groups": list(assignment[name]),
        })

    return {
        "requested_k": k,
        "k": k_used,
        "degraded": k_used != k,
        "degradation_reason": (None if k_used == k else
                               f"k={k} cannot keep every class in every fold; using k={k_used}"),
        "folds": folds,
    }


def _largest_valid_kfold(rows, n_groups: int, k: int, seed: int, labels):
    """The largest fold count <= k (and <= the number of groups) whose folds all keep every class."""
    for candidate in range(min(k, n_groups), 1, -1):
        base, extra = divmod(n_groups, candidate)
        sizes = {f"fold{index + 1}": base + (1 if index < extra else 0) for index in range(candidate)}
        try:
            _, assignment = stratified_group_partition(rows, sizes, seed=seed, labels=labels)
        except ValueError:
            continue
        return candidate, assignment
    raise ValueError("Cannot build two group-disjoint folds that each contain every class")


def _by_group(rows: list[dict]) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        groups[request_group(row)].append(row)
    return dict(groups)


def _cut(order: list[str], sizes: dict[str, int]) -> dict[str, list[str]]:
    """Consecutive slices of ``order``, one per part, of the requested sizes."""
    assignment, start = {}, 0
    for name, count in sizes.items():
        assignment[name] = order[start:start + count]
        start += count
    return assignment


def _stratification_error(parts, counts, total, fractions, n_rows, labels) -> float:
    """How far each part's row share and per-class share are from its target fraction."""
    score = sum(abs(len(parts[name]) / n_rows - fractions[name]) for name in parts)
    score += sum(abs(counts[name][label] / total[label] - fractions[name]) for name in parts for label in labels)
    return score
