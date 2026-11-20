"""``rules_plus``: the rules plus checks that remember the training rows.

(Logistic regression lives in ``logreg_baseline/`` and kNN in ``knn_baseline/``.) Anything remembered
comes from the training rows only.
"""
from __future__ import annotations

import numpy as np
from .checks import balance_unseen, empty_transfer, fit_balance_memory, parse_state, rule_cause
from .common import LABELS

RULE_CONFIDENCE = 0.97


# ---------------------------------------------------------------- rules + what the training fold remembers

def rule_cause_plus(state: str, memory: dict) -> str:
    """``rule_cause`` plus two checks: a transfer that returned nothing, and a balance never seen for that account."""
    parsed = parse_state(state)
    if empty_transfer(parsed):
        return "adversary"
    cause = rule_cause(state)
    if cause == "none" and balance_unseen(parsed, memory):
        return "adversary"
    return cause


def rule_probabilities(rows, cause_for) -> np.ndarray:
    cause = np.full((len(rows), len(LABELS)), (1 - RULE_CONFIDENCE) / (len(LABELS) - 1))
    for index, row in enumerate(rows):
        cause[index, LABELS.index(cause_for(row["state"]))] = RULE_CONFIDENCE
    return cause
