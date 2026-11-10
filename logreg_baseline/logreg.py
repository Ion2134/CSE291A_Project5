"""Multinomial logistic regression on the banking monitor's comparison features.

The features come from ``mlp_baseline.checks`` (call vs mandate, args vs call, output vs args and
ledger, ...). The model learns a weight per feature per cause, so it is a learned version of the
hand-written ``rules`` cascade and returns real probabilities. ``logreg_memory`` adds two features
that remember the training rows (see ``mlp_baseline/README.md``); read it as an upper bound.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from mlp_baseline.checks import (FEATURE_NAMES, apply_view, balance_unseen, check_matrix, empty_transfer,
                                 fit_balance_memory, parse_state)
from mlp_baseline.common import (LABELS, cause_of, check_monitor_row, input_id, read_jsonl, request_group,
                                 require_empty_dir, write_json, write_jsonl)
from mlp_baseline.metrics import evaluate

MODELS = ("logreg_checks", "logreg_memory")
MEMORY_FEATURES = ("balance_unseen", "empty_transfer_output")


def fit_predict(train: Path, test: Path, out: Path, *, model: str = "logreg_checks", view: str = "full",
                seed: int = 42, c: float = 1.0, validation: Path | None = None) -> dict:
    """Fit on ``train`` (+ ``validation``) and write predictions for ``test`` in the tester's format."""
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    out = Path(out)
    require_empty_dir(out, "Use a new or empty output directory")
    started = time.perf_counter()

    train_rows = read_jsonl(Path(train)) + (read_jsonl(Path(validation)) if validation is not None else [])
    test_rows = read_jsonl(Path(test))
    _check_fold(train_rows, test_rows)
    test_ids = [input_id(row) for row in test_rows]  # Ids stay those of the files the tester wrote.
    train_rows, test_rows = apply_view(train_rows, view), apply_view(test_rows, view)

    cause, details = logreg_probabilities(train_rows, test_rows, memory=model == "logreg_memory", seed=seed, c=c)

    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "test_predictions.jsonl", [
        {"input_id": row_id, "expected": row["expected"],
         "cause_probabilities": dict(zip(LABELS, map(float, probs))), "correct_probability": float(probs[0])}
        for row_id, row, probs in zip(test_ids, test_rows, cause)])

    result = {"model": model, "view": view, "seed": seed, "train_rows": len(train_rows),
              "test_rows": len(test_rows), "details": {"kind": model, **details},
              "test": evaluate(test_rows, cause, cause[:, 0]), "elapsed_seconds": time.perf_counter() - started}
    write_json(out / "results.json", result)
    return result


def logreg_probabilities(train_rows, test_rows, *, memory: bool = False, seed: int = 42, c: float = 1.0):
    """Cause probabilities in ``LABELS`` order, plus the learned weights.

    With ``memory``, two extra features are added, both built from the training rows only.
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


def _memory_columns(states, memory: dict) -> np.ndarray:
    parsed = [parse_state(state) for state in states]
    return np.array([[balance_unseen(p, memory), empty_transfer(p)] for p in parsed],
                    dtype=np.float32).reshape(-1, 2)


def _out_of_group_memory_columns(train_rows) -> np.ndarray:
    """Memory columns for training rows, each computed from the other request groups only.

    Memory is built from ``none`` rows, so a row looked up in its own memory could never fire the
    feature; this keeps the training columns distributed like the test columns.
    """
    groups = [request_group(row) for row in train_rows]
    columns = np.zeros((len(train_rows), 2), dtype=np.float32)
    for group in set(groups):
        inside = [i for i, g in enumerate(groups) if g == group]
        memory = fit_balance_memory([row for i, row in enumerate(train_rows) if groups[i] != group])
        columns[inside] = _memory_columns([train_rows[i]["state"] for i in inside], memory)
    return columns


def _check_fold(train_rows, test_rows) -> None:
    """Both parts are non-empty, share one schema, have valid labels, and share no request group."""
    if not train_rows or not test_rows:
        raise ValueError("Train and test files must not be empty")
    questions = train_rows[0]["questions"]
    for name, rows in (("train", train_rows), ("test", test_rows)):
        for line, row in enumerate(rows, 1):
            check_monitor_row(row, name, line, questions)
    if {request_group(row) for row in train_rows} & {request_group(row) for row in test_rows}:
        raise ValueError("A user-request group appears in both train and test")
