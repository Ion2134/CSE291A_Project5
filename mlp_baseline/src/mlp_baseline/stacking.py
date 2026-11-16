"""The ``stacking`` monitor: the frozen injection detector as a feature, plus a first-party stacker.

The injection detector (trained on public data) is frozen and only supplies P(injection). A
multinomial logistic regression learns the three causes from that probability plus TF-IDF
features of first-party rows. Stacking, rather than thresholding P(injection), lets first-party
labels teach that a resisted attack (high injection signal, correct output) is ``none``.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .checks import check_matrix
from .common import LABELS, cause_of, read_json, write_json
from .injection_model import INJECTION_INDEX, InjectionMLPPredictor, load_injection_mlp
from .metrics import apply_calibration, scale_temperature
from .mlp import CHECK_SCALE, CONFIG_FILE, MONITOR_FILE, NGRAM_RANGE

KIND = "stacked_logreg"
STACKER_FILE = "stacker.pkl"
FEATURES = "calibrated injection probability + TF-IDF (fit on first-party train pool only)"


@dataclass
class StackedMonitor:
    """Three-class cause monitor built on a frozen binary injection model."""

    frozen: InjectionMLPPredictor
    vectorizer: TfidfVectorizer
    stacker: LogisticRegression
    calibration: dict
    use_checks: bool = False

    def cause_proba(self, rows):
        """Cause probabilities with only the cause temperature applied."""
        return scale_temperature(self._raw_proba(rows), self.calibration["cause"]["temperature"])

    def predict(self, rows):
        """Calibrated ``(cause probabilities, P(correct))``; P(correct) comes from P(cause=none)."""
        return apply_calibration(self._raw_proba(rows), self.calibration)

    def _raw_proba(self, rows):
        values = self.stacker.predict_proba(stacker_features(rows, self.frozen, self.vectorizer, self.use_checks))
        return values[:, np.argsort(self.stacker.classes_)]  # Columns in LABELS order.


def stacker_features(rows, frozen, vectorizer, use_checks=False):
    """``[P(injection), TF-IDF..., comparison features]`` per row (the last block only with ``use_checks``)."""
    injection = frozen.predict_proba(rows)[:, INJECTION_INDEX]
    texts = [row["state"] for row in rows]
    columns = [injection, vectorizer.transform(texts).toarray()]
    if use_checks:
        columns.append(CHECK_SCALE * check_matrix(texts))
    return np.column_stack(columns)


def fit_stacker(rows, frozen, *, seed=42, max_features=4096, C=1.0, max_iter=2000, use_checks=False):
    """Fit the TF-IDF vectorizer and logistic stacker on first-party rows only."""
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=NGRAM_RANGE, dtype=np.float32)
    vectorizer.fit([row["state"] for row in rows])

    targets = np.array([LABELS.index(cause_of(row)) for row in rows])
    stacker = LogisticRegression(C=C, class_weight="balanced", max_iter=max_iter, random_state=seed)
    stacker.fit(stacker_features(rows, frozen, vectorizer, use_checks), targets)
    if set(stacker.classes_) != set(range(len(LABELS))):
        raise ValueError("Stacker training rows must include every cause class")
    return vectorizer, stacker


def save_stacker(path: Path, vectorizer, stacker, calibration, *, base_checkpoint: str, training: dict,
                 use_checks: bool = False) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / STACKER_FILE).open("wb") as stream:
        pickle.dump({"vectorizer": vectorizer, "stacker": stacker}, stream)
    write_json(path / MONITOR_FILE, {"kind": KIND, "task": "banking_monitor", "labels": list(LABELS),
                                     "calibration": calibration, "base_checkpoint": base_checkpoint,
                                     "features": FEATURES, "use_checks": use_checks})
    write_json(path / CONFIG_FILE, training)


def load_stacker(path: Path) -> StackedMonitor:
    path = Path(path)
    metadata = read_json(path / MONITOR_FILE)
    if metadata["kind"] != KIND or metadata["labels"] != list(LABELS):
        raise ValueError("Checkpoint is not a stacked banking monitor")

    with (path / STACKER_FILE).open("rb") as stream:
        saved = pickle.load(stream)
    frozen = load_injection_mlp(Path(metadata["base_checkpoint"]))
    return StackedMonitor(frozen, saved["vectorizer"], saved["stacker"], metadata["calibration"],
                          metadata.get("use_checks", False))
