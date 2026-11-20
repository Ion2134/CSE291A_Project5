"""The monitors as plug-in classifiers, so the simulator can ask them about each tool call as it runs.

Every classifier has the same two methods (dependency injection: the simulator only knows this shape):

    fit(train_rows) -> self          # no-op for the ones that need no training
    predict(rows)   -> (n, 3) array   # cause probabilities in LABELS order: none, agent, adversary

Rows are the monitor format the rest of the project uses (``{"state", "expected", "questions"}``, as in
``laya_train.jsonl``). Every classifier except ``oracle`` sees the **middleware view** of the state (no
``ledger_snapshot`` or ``received_args``), unless its spec says ``"view": "full"``.

A run lists classifiers by spec: a name, or a dict with the name and that classifier's inputs::

    ["rules", {"name": "logreg_checks", "train": "path/train.jsonl"}, {"name": "mlp", "checkpoint": "path/model"}]

``Panel(specs)`` builds and fits them once; ``panel.classify(row)`` returns every output, and
``decide(outputs)`` says whether recovery should run (more than half of the classifiers flag the call).
To add a classifier, write a class with ``fit``/``predict`` and add it to ``CLASSIFIERS``.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from mlp_baseline.checks import apply_view, fit_balance_memory, rule_cause
from mlp_baseline.common import LABELS, cause_of, read_jsonl
from mlp_baseline.variants import rule_cause_plus, rule_probabilities

ROOT = Path(__file__).resolve().parent.parent  # the repo root; relative paths in specs start here
# logreg and kNN are script folders, not installed packages.
for _folder in ("logreg_baseline", "knn_baseline"):
    if str(ROOT / "classifiers" / _folder) not in sys.path:
        sys.path.append(str(ROOT / "classifiers" / _folder))


class Classifier(Protocol):
    def fit(self, train_rows: list[dict]) -> "Classifier": ...
    def predict(self, rows: list[dict]) -> np.ndarray: ...


# ---------------------------------------------------------------- the classifiers

@dataclass
class Oracle:
    """The true label as a certain prediction: a perfect classifier (the recovery upper bound)."""

    def fit(self, train_rows):
        return self

    def predict(self, rows):
        return np.eye(len(LABELS))[[LABELS.index(cause_of(row)) for row in rows]]


@dataclass
class Rules:
    """The hand-written checks cascade (``mlp_baseline.checks.rule_cause``)."""

    def fit(self, train_rows):
        return self

    def predict(self, rows):
        return rule_probabilities(rows, rule_cause)


@dataclass
class RulesPlus:
    """The rules plus checks that remember the training rows' balances."""

    memory: dict = field(default_factory=dict)

    def fit(self, train_rows):
        self.memory = fit_balance_memory(train_rows)
        return self

    def predict(self, rows):
        return rule_probabilities(rows, lambda state: rule_cause_plus(state, self.memory))


@dataclass
class Logreg:
    """``logreg_checks`` (``memory=False``) or ``logreg_memory``; extra spec keys go to ``fit_logreg``."""

    memory: bool = False
    options: dict = field(default_factory=dict)
    model: Any = None

    def fit(self, train_rows):
        from logreg import fit_logreg
        self.model = fit_logreg(train_rows, memory=self.memory, **self.options)
        return self

    def predict(self, rows):
        return self.model.predict(rows)


@dataclass
class Knn:
    """``knn`` (TF-IDF) or ``knn_embed``; extra spec keys go to ``fit_knn`` (default: checks on, 5 neighbours)."""

    embedding: bool = False
    options: dict = field(default_factory=dict)
    model: Any = None

    def fit(self, train_rows):
        from knn import fit_knn
        from mlp_baseline.embeddings import DEFAULT_EMBEDDING
        options = {"use_checks": True, "neighbors": 5, **self.options}
        if self.embedding:
            options.setdefault("embedding", DEFAULT_EMBEDDING)
        self.model = fit_knn(train_rows, **options)
        return self

    def predict(self, rows):
        return self.model.predict(rows)


@dataclass
class Checkpoint:
    """``mlp`` / ``mlp_embed`` / ``stacking``: load a model saved by ``mlp-baseline`` instead of training."""

    kind: str
    checkpoint: Path | None = None
    model: Any = None

    def fit(self, train_rows):
        if self.checkpoint is None:
            raise ValueError(f"{self.kind} needs a 'checkpoint' (a saved model folder) in its spec")
        if self.kind == "stacking":
            from mlp_baseline.stacking import load_stacker
            self.model = load_stacker(self.checkpoint)
        else:
            from mlp_baseline.mlp import load_mlp
            self.model = load_mlp(self.checkpoint)
        return self

    def predict(self, rows):
        if self.kind == "stacking":
            return self.model.predict(rows)[0]
        return self.model.predict_proba(rows)


# name -> (factory taking the spec's extra keys, does it need training rows?)
CLASSIFIERS: dict[str, tuple[Callable[..., Classifier], bool]] = {
    "oracle": (lambda **_: Oracle(), False),
    "rules": (lambda **_: Rules(), False),
    "rules_plus": (lambda **_: RulesPlus(), True),
    "logreg_checks": (lambda **options: Logreg(memory=False, options=options), True),
    "logreg_memory": (lambda **options: Logreg(memory=True, options=options), True),
    "knn": (lambda **options: Knn(embedding=False, options=options), True),
    "knn_embed": (lambda **options: Knn(embedding=True, options=options), True),
    "mlp": (lambda checkpoint=None, **_: Checkpoint("mlp", _optional_path(checkpoint)), False),
    "mlp_embed": (lambda checkpoint=None, **_: Checkpoint("mlp_embed", _optional_path(checkpoint)), False),
    "stacking": (lambda checkpoint=None, **_: Checkpoint("stacking", _optional_path(checkpoint)), False),
}
SPEC_KEYS = ("name", "label", "train", "view")  # every other key is passed to the classifier


# ---------------------------------------------------------------- building a panel and deciding

@dataclass
class Member:
    label: str  # the key its outputs are recorded under (the name, unless the spec sets "label")
    classifier: Classifier
    view: str   # "middleware" or "full"; oracle always reads the label, so its view is irrelevant


class Panel:
    """The configured classifiers, built and fitted once; ``classify`` scores one monitor row with each."""

    def __init__(self, specs) -> None:
        self.members = [build(spec) for spec in specs]
        labels = [member.label for member in self.members]
        if len(set(labels)) != len(labels):
            raise ValueError(f"classifier labels must be unique, got {labels}; set 'label' in the spec")

    def classify(self, row: dict) -> dict[str, dict[str, float]]:
        outputs = {}
        for member in self.members:
            probabilities = member.classifier.predict(apply_view([row], member.view))[0]
            outputs[member.label] = {label: round(float(p), 6) for label, p in zip(LABELS, probabilities)}
        return outputs


def build(spec) -> Member:
    """One spec (a name, or a dict with ``name`` and that classifier's inputs) -> a fitted classifier."""
    spec = {"name": spec} if isinstance(spec, str) else dict(spec)
    name = spec.get("name")
    if name not in CLASSIFIERS:
        raise ValueError(f"unknown classifier {name!r}; choose from {sorted(CLASSIFIERS)}")
    factory, needs_training = CLASSIFIERS[name]
    view = spec.get("view", "middleware")
    train_rows = []
    if needs_training:
        if "train" not in spec:
            raise ValueError(f"{name} needs 'train' (a monitor-format .jsonl of training rows) in its spec")
        train_rows = apply_view(read_jsonl(_path(spec["train"])), view)
    classifier = factory(**{key: value for key, value in spec.items() if key not in SPEC_KEYS}).fit(train_rows)
    return Member(spec.get("label", name), classifier, view)


def top_cause(probabilities: dict[str, float]) -> str:
    """The most likely cause (ties go to the earlier label: none, agent, adversary)."""
    return max(probabilities.items(), key=lambda item: item[1])[0]


def decide(outputs: dict[str, dict[str, float]]) -> dict:
    """Recover when more than half of the classifiers flag the call (their top cause is not ``none``)."""
    flagged = [label for label, probabilities in outputs.items() if top_cause(probabilities) != "none"]
    return {"recover": bool(outputs) and len(flagged) * 2 > len(outputs), "flagged_by": flagged}


def _path(value: str | Path) -> Path:
    """A spec path: absolute as given, otherwise relative to the repo root."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _optional_path(value: str | Path | None) -> Path | None:
    return None if value is None else _path(value)
