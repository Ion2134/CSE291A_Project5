"""Gradient-boosted trees on the same features as the logistic-regression monitor.

``gbm_checks`` and ``gbm_memory`` use exactly the inputs of ``logreg_checks`` and ``logreg_memory``
(the 12 comparison flags, the 4 hard-task flags, and for ``_memory`` the two memory features), built
by ``../logreg_baseline/logreg.py``. Only the learner changes: trees can combine flags, e.g. "the
amount was described *and* it differs from the mandate", which a linear model can only add up.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from mlp_baseline.checks import apply_view, fit_balance_memory
from mlp_baseline.common import LABELS, cause_of, input_id, read_jsonl, require_empty_dir, write_json, write_jsonl
from mlp_baseline.metrics import evaluate

from logreg import (HARD_FEATURES, MEMORY_FEATURES, _base_features, _check_fold, _memory_columns,
                    _out_of_group_memory_columns, fit_detector)
from mlp_baseline.checks import FEATURE_NAMES

MODELS = ("gbm_checks", "gbm_memory")

# Fixed, untuned settings for ~1-2k rows: shallow trees, a slow learning rate and a minimum leaf
# size so a handful of rows can't carve out their own leaf. No early stopping, so a fit is
# deterministic for a seed and uses every training row.
DEFAULT_PARAMS = {"max_iter": 200, "learning_rate": 0.05, "max_depth": 3, "max_leaf_nodes": 8,
                  "min_samples_leaf": 10, "l2_regularization": 1.0}


def fit_predict(train: Path, test: Path, out: Path, *, model: str = "gbm_checks", view: str = "full",
                seed: int = 42, validation: Path | None = None, injection_detector: str = "keyword",
                params: dict | None = None) -> dict:
    """Fit on ``train`` (+ ``validation``) and write predictions for ``test`` in the tester's format."""
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    out = Path(out)
    require_empty_dir(out, "Use a new or empty output directory")
    started = time.perf_counter()

    train_rows = read_jsonl(Path(train)) + (read_jsonl(Path(validation)) if validation is not None else [])
    test_rows = read_jsonl(Path(test))
    _check_fold(train_rows, test_rows)
    test_ids = [input_id(row) for row in test_rows]
    train_rows, test_rows = apply_view(train_rows, view), apply_view(test_rows, view)

    fitted = fit_gbm(train_rows, memory=model == "gbm_memory", seed=seed,
                     injection_detector=injection_detector, params=params)
    cause = fitted.predict(test_rows)

    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "test_predictions.jsonl", [
        {"input_id": row_id, "expected": row["expected"],
         "cause_probabilities": dict(zip(LABELS, map(float, probs))), "correct_probability": float(probs[0])}
        for row_id, row, probs in zip(test_ids, test_rows, cause)])

    result = {"model": model, "view": view, "seed": seed, "train_rows": len(train_rows),
              "test_rows": len(test_rows), "details": {"kind": model, **fitted.details()},
              "test": evaluate(test_rows, cause, cause[:, 0]), "elapsed_seconds": time.perf_counter() - started}
    write_json(out / "results.json", result)
    return result


@dataclass
class GbmModel:
    """A fitted GBM monitor: score any rows with ``predict`` (what ``recovery_classifiers`` calls)."""

    model: HistGradientBoostingClassifier
    detector: object
    memory: dict | None
    names: list[str]
    injection_detector: str
    params: dict = field(default_factory=dict)

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
        return {"features": self.names, "params": self.params, "injection_detector": self.injection_detector,
                "trees_per_class": int(self.model.n_iter_)}


def fit_gbm(train_rows, *, memory: bool = False, seed: int = 42, injection_detector: str = "keyword",
            params: dict | None = None) -> GbmModel:
    """Fit on ``train_rows``; features are built exactly as for ``fit_logreg``."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    states = [row["state"] for row in train_rows]
    detector = fit_detector(train_rows, injection_detector)
    x_train = _base_features(states, detector)
    names = [*FEATURE_NAMES, *HARD_FEATURES]
    if memory:
        x_train = np.hstack([x_train, _out_of_group_memory_columns(train_rows)])
        names += list(MEMORY_FEATURES)
    y_train = np.array([LABELS.index(cause_of(row)) for row in train_rows])
    # Same class balancing as logreg (class_weight="balanced"), so the two are compared like for like.
    model = HistGradientBoostingClassifier(class_weight="balanced", early_stopping=False, random_state=seed,
                                           **params).fit(x_train, y_train)
    return GbmModel(model, detector, fit_balance_memory(train_rows) if memory else None, names,
                    injection_detector, params)
