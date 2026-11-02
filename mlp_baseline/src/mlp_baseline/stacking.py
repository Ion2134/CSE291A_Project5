"""Hybrid monitor: frozen external injection model + first-party stacker.

The binary injection MLP (trained on pinned external datasets) is frozen and
used only as a feature generator. A first-party multinomial logistic
regression learns the three-way cause mapping from the calibrated injection
probability plus TF-IDF features. Stacking — rather than threshold mapping —
lets first-party labels teach the stacker that a resisted attack (injection
signal high, output still correct) is ``none``, not ``adversary``.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .data import LABELS, write_json
from .injection_mlp import load_injection_mlp
from .metrics import scale_temperature


def stacker_features(rows, frozen, vectorizer):
    injection = frozen.predict_proba(rows)[:, 1]
    tfidf = vectorizer.transform([row["state"] for row in rows]).toarray()
    return np.column_stack([injection, tfidf])


def fit_stacker(rows, frozen, *, seed=42, max_features=4096, C=1.0, max_iter=2000):
    """Fit the TF-IDF vectorizer and logistic stacker on first-party rows only."""
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), dtype=np.float32)
    vectorizer.fit([row["state"] for row in rows])
    targets = np.array([LABELS.index(row["expected"]["cause"]) for row in rows])
    stacker = LogisticRegression(C=C, class_weight="balanced", max_iter=max_iter, random_state=seed)
    stacker.fit(stacker_features(rows, frozen, vectorizer), targets)
    if set(stacker.classes_) != set(range(len(LABELS))):
        raise ValueError("Stacker training rows must include every cause class")
    return vectorizer, stacker


class StackedMonitor:
    """Three-class cause monitor built on a frozen binary injection model."""

    def __init__(self, frozen, vectorizer, stacker, calibration):
        self.frozen, self.vectorizer, self.stacker = frozen, vectorizer, stacker
        self.calibration = calibration

    def cause_proba(self, rows):
        values = self.stacker.predict_proba(stacker_features(rows, self.frozen, self.vectorizer))
        order = np.argsort(self.stacker.classes_)
        values = values[:, order]
        return scale_temperature(values, self.calibration["cause"]["temperature"])

    def predict(self, rows):
        cause = self.cause_proba(rows)
        correct = cause[:, 0]  # The three cause classes form a partition; correctness is P(cause=none).
        binary = scale_temperature(np.stack([1 - correct, correct], axis=1),
                                   self.calibration["correctness"]["temperature"])
        return cause, binary[:, 1]


def save_stacker(path: Path, vectorizer, stacker, calibration, *, base_checkpoint: str, training: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "stacker.pkl").open("wb") as stream:
        pickle.dump({"vectorizer": vectorizer, "stacker": stacker}, stream)
    write_json(path / "monitor.json", {
        "kind": "stacked_logreg",
        "task": "banking_monitor",
        "labels": list(LABELS),
        "calibration": calibration,
        "base_checkpoint": base_checkpoint,
        "features": "calibrated injection probability + TF-IDF (fit on first-party train pool only)",
    })
    write_json(path / "model_config.json", training)


def load_stacker(path: Path):
    metadata = json.loads((Path(path) / "monitor.json").read_text())
    if metadata["kind"] != "stacked_logreg" or metadata["labels"] != list(LABELS):
        raise ValueError("Checkpoint is not a stacked banking monitor")
    with (Path(path) / "stacker.pkl").open("rb") as stream:
        saved = pickle.load(stream)
    frozen = load_injection_mlp(Path(metadata["base_checkpoint"]))
    return StackedMonitor(frozen, saved["vectorizer"], saved["stacker"], metadata["calibration"])
