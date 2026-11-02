"""Group-disjoint stratified partitioning and cross-validation folds.

Splits never separate rows that share a normalized user request, and every
part keeps every cause class whenever the data allows it.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict

from .data import LABELS, request_group


def _by_group(rows: list[dict]) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        groups[request_group(row)].append(row)
    return dict(groups)


def stratified_group_partition(rows: list[dict], sizes: dict[str, int], *, seed: int = 42,
                               attempts: int = 4000, labels=LABELS):
    """Partition rows into named group-disjoint parts with stratified labels.

    ``sizes`` maps part names to the number of request groups each receives.
    Returns ``(parts, assignment)`` where ``parts`` maps names to rows and
    ``assignment`` maps names to the request-group keys that form the part.
    Every part must contain every label; raises ValueError when no assignment
    satisfies that within ``attempts`` shuffles.
    """
    groups = _by_group(rows)
    keys = sorted(groups)
    if len(sizes) < 2 or min(sizes.values()) < 1 or sum(sizes.values()) != len(keys):
        raise ValueError("Part sizes must be positive and account for every request group")
    total = Counter(row["expected"]["cause"] for row in rows)
    if set(total) != set(labels):
        raise ValueError("The dataset must include every cause class")
    fractions = {name: count / len(keys) for name, count in sizes.items()}
    rng = random.Random(seed)
    best = None
    for _ in range(attempts):
        order = keys.copy()
        rng.shuffle(order)
        assignment, start = {}, 0
        for name, count in sizes.items():
            assignment[name] = order[start:start + count]
            start += count
        candidate = {name: [row for key in selected for row in groups[key]]
                     for name, selected in assignment.items()}
        counts = {name: Counter(row["expected"]["cause"] for row in part)
                  for name, part in candidate.items()}
        # Hard constraint: no part may lose a class; stratification is scored below.
        if any(set(counts[name]) != set(labels) for name in candidate):
            continue
        score = sum(abs(len(candidate[name]) / len(rows) - fractions[name]) for name in candidate)
        score += sum(abs(counts[name][label] / total[label] - fractions[name])
                     for name in candidate for label in labels)
        if best is None or score < best[0]:
            best = (score, candidate, assignment)
    if best is None:
        raise ValueError("No group-disjoint partition keeps every class in every part")
    _, parts, assignment = best
    return parts, assignment


def group_kfold(rows: list[dict], k: int = 5, *, seed: int = 42, labels=LABELS) -> dict:
    """Build k group-disjoint stratified cross-validation folds.

    Starts from the requested ``k`` and degrades toward 2 folds until every
    fold can hold every class; the chosen ``k`` and any degradation reason
    are recorded in the result.
    """
    groups = _by_group(rows)
    if len(groups) < 2:
        raise ValueError("Need at least two request groups for cross-validation")
    if k < 2:
        raise ValueError("k must be at least 2")
    chosen = None
    for candidate in range(min(k, len(groups)), 1, -1):
        base, extra = divmod(len(groups), candidate)
        sizes = {f"fold{index + 1}": base + (1 if index < extra else 0) for index in range(candidate)}
        try:
            _, assignment = stratified_group_partition(rows, sizes, seed=seed, labels=labels)
        except ValueError:
            continue
        chosen = (candidate, assignment)
        break
    if chosen is None:
        raise ValueError("Cannot build two group-disjoint folds that each contain every class")
    k_used, assignment = chosen
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
