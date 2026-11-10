"""Extra monitors that share the fit-predict interface: rules with training-fold memory, and k-nearest-neighbours.

Everything here sees the same (possibly middleware-view) rows as the other models, and anything it
"learns" comes from the training rows only.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from sklearn.linear_model import LogisticRegression

from .checks import FEATURE_NAMES, check_matrix, parse_state, rule_cause
from .common import LABELS, cause_of, request_group
from .mlp import NGRAM_RANGE, TrainSettings, _dense

RULE_CONFIDENCE = 0.97
SMOOTHING = 0.05


# ---------------------------------------------------------------- rules + what the training fold remembers

def fit_balance_memory(train_rows) -> dict:
    """account -> the set of balances its ``get_balance`` outputs showed in training rows labelled ``none``."""
    seen = defaultdict(set)
    for row in train_rows:
        parsed = parse_state(row["state"])
        if parsed and parsed.get("tool") == "get_balance" and cause_of(row) == "none":
            output = parsed["output"] if isinstance(parsed["output"], dict) else {}
            if "balance_cents" in output:
                seen[output.get("account_id")].add(output["balance_cents"])
    return dict(seen)


def _empty_transfer(parsed) -> bool:
    """A finished transfer call whose result is empty (a call that has not run yet has no output line)."""
    return bool(parsed and parsed["has_output"] and parsed.get("tool") == "initiate_transfer"
                and parsed["output"] == {})


def _balance_unseen(parsed, memory: dict) -> bool:
    """A first-read balance that training never showed for that (known) account."""
    if not parsed or parsed.get("tool") != "get_balance" or parsed["prior_transfer"]:
        return False
    output = parsed["output"]
    if not isinstance(output, dict) or "balance_cents" not in output:
        return False
    known = memory.get(output.get("account_id"))
    return bool(known) and output["balance_cents"] not in known


def rule_cause_plus(state: str, memory: dict) -> str:
    """``rule_cause`` plus two checks: a transfer that returned nothing, and a balance never seen for that account."""
    parsed = parse_state(state)
    if _empty_transfer(parsed):
        return "adversary"
    cause = rule_cause(state)
    if cause == "none" and _balance_unseen(parsed, memory):
        return "adversary"
    return cause


def rule_probabilities(rows, cause_for) -> np.ndarray:
    cause = np.full((len(rows), len(LABELS)), (1 - RULE_CONFIDENCE) / (len(LABELS) - 1))
    for index, row in enumerate(rows):
        cause[index, LABELS.index(cause_for(row["state"]))] = RULE_CONFIDENCE
    return cause


# ---------------------------------------------------------------- k-nearest-neighbours

def knn_probabilities(train_rows, test_rows, settings: TrainSettings, *, neighbors: int = 5) -> np.ndarray:
    """Cosine kNN on the MLP's own input features; votes are weighted by closeness and by 1 / class frequency."""
    embedding = settings.embedding
    vectorizer = None if embedding else TfidfVectorizer(max_features=settings.max_features,
                                                        ngram_range=NGRAM_RANGE, dtype=np.float32)
    x_train = _dense(vectorizer, train_rows, settings.use_checks, fit=True, embedding=embedding)
    x_test = _dense(vectorizer, test_rows, settings.use_checks, fit=False, embedding=embedding)
    y_train = np.array([LABELS.index(cause_of(row)) for row in train_rows])
    class_weight = len(y_train) / (len(LABELS) * np.maximum(np.bincount(y_train, minlength=len(LABELS)), 1))

    index = NearestNeighbors(n_neighbors=min(neighbors, len(train_rows)), metric="cosine").fit(x_train)
    distance, neighbor = index.kneighbors(x_test)
    votes = np.zeros((len(test_rows), len(LABELS)))
    for row in range(len(test_rows)):
        for dist, other in zip(distance[row], neighbor[row]):
            votes[row, y_train[other]] += class_weight[y_train[other]] / (dist + 1e-3)
    votes += SMOOTHING * votes.sum(axis=1, keepdims=True)
    return votes / votes.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- logistic regression on the comparison features

MEMORY_FEATURES = ("balance_unseen", "empty_transfer_output")


def _memory_columns(states, memory: dict) -> np.ndarray:
    parsed = [parse_state(state) for state in states]
    return np.array([[_balance_unseen(p, memory), _empty_transfer(p)] for p in parsed], dtype=np.float32).reshape(-1, 2)


def _out_of_group_memory_columns(train_rows) -> np.ndarray:
    """Memory columns for training rows, each computed from the other request groups only.

    Memory is built from ``none`` rows, so a row seen by its own memory could never fire the feature;
    this keeps the training columns distributed like the test columns.
    """
    groups = [request_group(row) for row in train_rows]
    columns = np.zeros((len(train_rows), 2), dtype=np.float32)
    for group in set(groups):
        inside = [i for i, g in enumerate(groups) if g == group]
        memory = fit_balance_memory([row for i, row in enumerate(train_rows) if groups[i] != group])
        columns[inside] = _memory_columns([train_rows[i]["state"] for i in inside], memory)
    return columns


def logreg_probabilities(train_rows, test_rows, *, memory: bool = False, seed: int = 42, c: float = 1.0):
    """Multinomial logistic regression on the 0/1 comparison features (plus the memory features when ``memory``).

    Returns ``(probabilities in LABELS order, details with the learned weights)``.
    """
    x_train = check_matrix([row["state"] for row in train_rows])
    x_test = check_matrix([row["state"] for row in test_rows])
    names = list(FEATURE_NAMES)
    if memory:
        x_train = np.hstack([x_train, _out_of_group_memory_columns(train_rows)])
        x_test = np.hstack([x_test, _memory_columns([row["state"] for row in test_rows],
                                                    fit_balance_memory(train_rows))])
        names += list(MEMORY_FEATURES)

    y_train = np.array([LABELS.index(cause_of(row)) for row in train_rows])
    model = LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=seed).fit(x_train, y_train)
    probabilities = np.zeros((len(test_rows), len(LABELS)))
    probabilities[:, model.classes_] = model.predict_proba(x_test)

    weights = {LABELS[label]: {name: round(float(w), 3) for name, w in zip(names, row)}
               for label, row in zip(model.classes_, model.coef_)} if len(model.classes_) > 2 else {}
    return probabilities, {"features": names, "C": c, "weights": weights}
