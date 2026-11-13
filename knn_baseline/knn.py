"""Cosine k-nearest-neighbours on the banking monitor's input features.

Features are TF-IDF (``knn``) or a sentence embedding (``knn_embed``) of the state text, optionally
plus the comparison features from ``mlp_baseline.checks`` (``--use-checks``), exactly as the MLP
sees them. A row's cause probabilities are a vote of its nearest training rows, weighted by
closeness and by 1 / class frequency so the rare causes are not outvoted.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from mlp_baseline.checks import apply_view, check_matrix
from mlp_baseline.common import (LABELS, cause_of, check_monitor_row, input_id, read_jsonl, request_group,
                                 require_empty_dir, write_json, write_jsonl)
from mlp_baseline.embeddings import DEFAULT_EMBEDDING, embed_texts
from mlp_baseline.metrics import evaluate
from mlp_baseline.mlp import CHECK_SCALE, NGRAM_RANGE  # The MLP's feature constants, so the inputs match.

MODELS = ("knn", "knn_embed")
SMOOTHING = 0.05


def fit_predict(train: Path, test: Path, out: Path, *, model: str = "knn", view: str = "full",
                use_checks: bool = False, neighbors: int = 5, max_features: int = 4096,
                embedding: str | None = None, validation: Path | None = None) -> dict:
    """Index ``train`` (+ ``validation``) and write predictions for ``test`` in the tester's format."""
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    if model == "knn_embed" and not embedding:
        embedding = DEFAULT_EMBEDDING
    out = Path(out)
    require_empty_dir(out, "Use a new or empty output directory")
    started = time.perf_counter()

    train_rows = read_jsonl(Path(train)) + (read_jsonl(Path(validation)) if validation is not None else [])
    test_rows = read_jsonl(Path(test))
    _check_fold(train_rows, test_rows)
    test_ids = [input_id(row) for row in test_rows]  # Ids stay those of the files the tester wrote.
    train_rows, test_rows = apply_view(train_rows, view), apply_view(test_rows, view)

    cause = knn_probabilities(train_rows, test_rows, neighbors=neighbors, use_checks=use_checks,
                              max_features=max_features, embedding=embedding)

    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "test_predictions.jsonl", [
        {"input_id": row_id, "expected": row["expected"],
         "cause_probabilities": dict(zip(LABELS, map(float, probs))), "correct_probability": float(probs[0])}
        for row_id, row, probs in zip(test_ids, test_rows, cause)])

    result = {"model": model, "view": view, "train_rows": len(train_rows), "test_rows": len(test_rows),
              "details": {"kind": model, "neighbors": neighbors, "embedding": embedding, "use_checks": use_checks},
              "test": evaluate(test_rows, cause, cause[:, 0]), "elapsed_seconds": time.perf_counter() - started}
    write_json(out / "results.json", result)
    return result


def knn_probabilities(train_rows, test_rows, *, neighbors: int = 5, use_checks: bool = False,
                      max_features: int = 4096, embedding: str | None = None) -> np.ndarray:
    """Cause probabilities in ``LABELS`` order; the index is built from ``train_rows`` only."""
    vectorizer = None if embedding else TfidfVectorizer(max_features=max_features, ngram_range=NGRAM_RANGE,
                                                        dtype=np.float32)
    x_train = _features(vectorizer, train_rows, use_checks, fit=True, embedding=embedding)
    x_test = _features(vectorizer, test_rows, use_checks, fit=False, embedding=embedding)
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


def _features(vectorizer, rows, use_checks: bool, *, fit: bool, embedding: str | None) -> np.ndarray:
    """TF-IDF or embedding matrix, plus scaled comparison features when ``use_checks`` (as in the MLP)."""
    texts = [row["state"] for row in rows]
    if embedding:
        base = embed_texts(texts, embedding)
    else:
        base = (vectorizer.fit_transform(texts) if fit else vectorizer.transform(texts)).toarray().astype(np.float32)
    return np.hstack([base, CHECK_SCALE * check_matrix(texts)]) if use_checks else base


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
