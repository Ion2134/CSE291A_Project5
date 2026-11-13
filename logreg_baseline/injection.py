"""Injection detectors: tool-output text -> P(injection) in [0, 1].

The score fills the ``output_injection_text`` feature of the logistic regression. Pick one with
``--injection-detector``:

- ``keyword``: the original regex from ``mlp_baseline.checks`` (0 or 1). The default.
- ``char_ngram``: character n-gram TF-IDF + logistic regression, fit on the training fold.

To add a detector, write a class with ``fit(texts, labels)`` and ``score(texts)`` and add it to ``DETECTORS``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from mlp_baseline.checks import _INJECTION, parse_state


class InjectionDetector(Protocol):
    def fit(self, texts: list[str], labels: list[int]) -> "InjectionDetector": ...

    def score(self, texts: list[str]) -> np.ndarray: ...


@dataclass
class KeywordDetector:
    """The original keyword regex; nothing to learn."""

    def fit(self, texts: list[str], labels: list[int]) -> "KeywordDetector":
        return self

    def score(self, texts: list[str]) -> np.ndarray:
        return np.array([float(bool(_INJECTION.search(text))) for text in texts], dtype=np.float32)


@dataclass
class CharNgramDetector:
    """Character 2-5-grams, so reworded or misspelled injections still share pieces with known ones."""

    vectorizer: TfidfVectorizer = field(
        default_factory=lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=True))
    model: LogisticRegression | None = None
    constant: float = 0.0   # Used when the training fold has only one class.

    def fit(self, texts: list[str], labels: list[int]) -> "CharNgramDetector":
        if len(set(labels)) < 2:
            self.constant = float(labels[0]) if labels else 0.0
            return self
        features = self.vectorizer.fit_transform(texts)
        self.model = LogisticRegression(class_weight="balanced", max_iter=2000)
        self.model.fit(features, labels)
        return self

    def score(self, texts: list[str]) -> np.ndarray:
        if self.model is None:
            return np.full(len(texts), self.constant, dtype=np.float32)
        return self.model.predict_proba(self.vectorizer.transform(texts))[:, 1].astype(np.float32)


DETECTORS: dict[str, type] = {"keyword": KeywordDetector, "char_ngram": CharNgramDetector}


def output_text(state: str) -> str:
    """The string values of the row's tool output, joined (the text the keyword regex always searched)."""
    output = _output(state)
    return " ".join(str(value) for value in output.values() if isinstance(value, str))


def was_injected(row: dict) -> int:
    """Training label: 1 if the row's output was injected.

    Merged rows don't record which attack ran, so this uses what every injection looks like today: an
    ``adversary`` row whose output has a private ``_`` key. Change this when rows carry the operator.
    """
    if row["expected"]["cause"] != "adversary":
        return 0
    return int(any(str(key).startswith("_") for key in _output(row["state"])))


def _output(state: str) -> dict:
    parsed = parse_state(state)
    output = parsed.get("output") if parsed else None
    return output if isinstance(output, dict) else {}
