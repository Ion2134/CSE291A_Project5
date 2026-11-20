"""Multinomial logistic regression on the banking monitor's comparison features.

The features come from ``mlp_baseline.checks`` (call vs mandate, args vs call, output vs args and
ledger, ...). The model learns a weight per feature per cause, so it is a learned version of the
hand-written ``rules`` cascade and returns real probabilities. Both models also get four
``HARD_FEATURES`` for mandates that describe a field instead of giving it. ``logreg_memory`` adds two features
that remember the training rows (see ``mlp_baseline/README.md``); read it as an upper bound.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from mlp_baseline.checks import (FEATURE_NAMES, apply_view, balance_unseen, check_matrix, empty_transfer,
                                 fit_balance_memory, parse_state)
from mlp_baseline.common import (LABELS, cause_of, check_monitor_row, input_id, read_jsonl, request_group,
                                 require_empty_dir, write_json, write_jsonl)
from mlp_baseline.metrics import evaluate

from injection import DETECTORS, output_text, was_injected

MODELS = ("logreg_checks", "logreg_memory")
MEMORY_FEATURES = ("balance_unseen", "empty_transfer_output")

# Hard tasks describe a field instead of giving it ("the account held by Gray Synth"), so the
# call-vs-mandate flags fire on correct calls. These say which fields were described, and whether
# the call's ids came from somewhere the agent could have seen them.
HARD_FEATURES = ("account_described", "payee_described", "amount_described", "call_id_unseen")
INJECTION_COLUMN = FEATURE_NAMES.index("output_injection_text")
_ID = re.compile(r"\b(?:acct|payee)_synth_\d+\b")
_ACCOUNT_KEYS = ("account_id", "from_account_id")


def fit_predict(train: Path, test: Path, out: Path, *, model: str = "logreg_checks", view: str = "full",
                seed: int = 42, c: float = 1.0, validation: Path | None = None,
                injection_detector: str = "keyword") -> dict:
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

    cause, details = logreg_probabilities(train_rows, test_rows, memory=model == "logreg_memory", seed=seed, c=c,
                                          injection_detector=injection_detector)

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


def logreg_probabilities(train_rows, test_rows, *, memory: bool = False, seed: int = 42, c: float = 1.0,
                         injection_detector: str = "keyword"):
    """Cause probabilities in ``LABELS`` order, plus the learned weights.

    With ``memory``, two extra features are added, both built from the training rows only.
    ``injection_detector`` (a ``DETECTORS`` name) fills the ``output_injection_text`` column.
    """
    fitted = fit_logreg(train_rows, memory=memory, seed=seed, c=c, injection_detector=injection_detector)
    return fitted.predict(test_rows), fitted.details()


@dataclass
class LogregModel:
    """A fitted logreg monitor: score any rows with ``predict`` (what ``classifiers`` calls per tool call)."""

    model: LogisticRegression
    detector: object
    memory: dict | None
    names: list[str]
    c: float
    injection_detector: str

    def predict(self, rows) -> np.ndarray:
        """Cause probabilities in ``LABELS`` order."""
        states = [row["state"] for row in rows]
        x = _base_features(states, self.detector)
        if self.memory is not None:
            x = np.hstack([x, _memory_columns(states, self.memory)])
        probabilities = np.zeros((len(rows), len(LABELS)))
        probabilities[:, self.model.classes_] = self.model.predict_proba(x)
        return probabilities

    def details(self) -> dict:
        weights = ({LABELS[label]: {name: round(float(w), 3) for name, w in zip(self.names, row)}
                    for label, row in zip(self.model.classes_, self.model.coef_)}
                   if len(self.model.classes_) > 2 else {})
        return {"features": self.names, "C": self.c, "injection_detector": self.injection_detector,
                "weights": weights}


def fit_logreg(train_rows, *, memory: bool = False, seed: int = 42, c: float = 1.0,
               injection_detector: str = "keyword") -> LogregModel:
    """Fit on ``train_rows``. Memory and the injection detector are built from the training rows only."""
    states = [row["state"] for row in train_rows]
    detector = fit_detector(train_rows, injection_detector)
    x_train = _base_features(states, detector)
    names = [*FEATURE_NAMES, *HARD_FEATURES]
    if memory:
        x_train = np.hstack([x_train, _out_of_group_memory_columns(train_rows)])
        names += list(MEMORY_FEATURES)
    y_train = np.array([LABELS.index(cause_of(row)) for row in train_rows])
    model = LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=seed).fit(x_train, y_train)
    return LogregModel(model, detector, fit_balance_memory(train_rows) if memory else None, names, c,
                       injection_detector)


def _base_features(states, detector) -> np.ndarray:
    """The comparison flags (injection column from ``detector``) plus the hard-task flags."""
    x = check_matrix(states)
    x[:, INJECTION_COLUMN] = detector.score([output_text(state) for state in states])
    return np.hstack([x, hard_matrix(states)])


def hard_features(state: str) -> np.ndarray:
    """``HARD_FEATURES``-ordered 0/1 vector for one state string (all zeros if it cannot be parsed).

    A field is *described* when the call sets it but the mandate has no literal value for it.
    ``call_id_unseen`` fires when an account or payee id in the call appears neither in the user
    message, the mandate, nor any earlier tool output, i.e. the agent could not have looked it up.
    """
    parsed = parse_state(state)
    if parsed is None:
        return np.zeros(len(HARD_FEATURES), dtype=np.float32)
    trusted, call = parsed["trusted"], parsed["model_call"]
    account_given = any(trusted.get(key) is not None for key in _ACCOUNT_KEYS)
    seen = set()
    for line in state.splitlines():
        if line.startswith(("user_message: ", "trusted: ")):
            seen.update(_ID.findall(line))
        elif line.lstrip().startswith("- [") and " output=" in line:
            seen.update(_ID.findall(line.split(" output=", 1)[1]))
    call_ids = set(_ID.findall(" ".join(str(value) for value in call.values())))
    return np.array([any(key in call for key in _ACCOUNT_KEYS) and not account_given,
                     "payee_id" in call and trusted.get("payee_id") is None,
                     "amount_cents" in call and trusted.get("amount_cents") is None,
                     bool(call_ids - seen)], dtype=np.float32)


def hard_matrix(states) -> np.ndarray:
    return np.array([hard_features(state) for state in states], dtype=np.float32).reshape(-1, len(HARD_FEATURES))


def injection_scores(train_rows, test_rows, name: str) -> tuple[np.ndarray, np.ndarray]:
    """P(injection) for the training and test rows, from a detector fit on the training rows only."""
    detector = fit_detector(train_rows, name)
    return (detector.score([output_text(row["state"]) for row in train_rows]),
            detector.score([output_text(row["state"]) for row in test_rows]))


def fit_detector(train_rows, name: str):
    """The ``name`` injection detector, fit on the training rows only."""
    if name not in DETECTORS:
        raise ValueError(f"injection_detector must be one of {sorted(DETECTORS)}")
    detector = DETECTORS[name]()
    detector.fit([output_text(row["state"]) for row in train_rows], [was_injected(row) for row in train_rows])
    return detector


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
