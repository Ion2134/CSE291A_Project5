"""The TF-IDF MLP shared by every model here: architecture, training loop, checkpoint save/load."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn

from .checks import check_matrix
from .common import LABELS, read_json, write_json

# Files in a checkpoint directory.
WEIGHTS_FILE = "model.pt"
VECTORIZER_FILE = "vectorizer.json"
CONFIG_FILE = "model_config.json"
MONITOR_FILE = "monitor.json"  # Task, labels and calibration; written by each pipeline.

NGRAM_RANGE = (1, 2)
SECOND_LAYER = 64
DROPOUT = 0.1
WEIGHT_DECAY = 0.01
CHECK_SCALE = 2.0  # Comparison features are 0/1; scaled so the L2-normalised TF-IDF block does not drown them.


@dataclass(frozen=True)
class TrainSettings:
    """Training knobs shared by every pipeline (each command sets its own defaults)."""

    seed: int = 42
    epochs: int = 40               # A maximum when ``patience`` is set.
    batch_size: int = 16
    max_features: int = 4096       # TF-IDF vocabulary size.
    patience: int | None = None    # Early-stopping patience in epochs (needs validation rows).
    threads: int = 4
    verbose: bool = False          # Print one line per epoch.
    use_checks: bool = False       # Append explicit comparison features (see checks.py) to the TF-IDF input.


class MLP(nn.Module):
    def __init__(self, inputs: int, hidden: int = 128, classes: int = len(LABELS)):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(), nn.Dropout(DROPOUT),
                                     nn.Linear(hidden, SECOND_LAYER), nn.ReLU(), nn.Linear(SECOND_LAYER, classes))

    def forward(self, values):
        return self.network(values)


class MLPPredictor:
    """A loaded checkpoint: vectorizer + network, in eval mode."""

    def __init__(self, model, vectorizer, device="cpu", use_checks=False):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.vectorizer = vectorizer
        self.use_checks = use_checks

    @torch.no_grad()
    def predict_proba(self, rows):
        values = _dense(self.vectorizer, rows, self.use_checks, fit=False)
        return self.model(torch.from_numpy(values).to(self.device)).softmax(-1).cpu().numpy()

    def predict(self, rows):
        """Cause probabilities and P(correct); the causes partition, so P(correct) = P(cause=none)."""
        result = self.predict_proba(rows)
        return result, result[:, 0]


# ---------------------------------------------------------------- training

def train_mlp(train, val, out: Path, settings: TrainSettings = TrainSettings(), *, hidden=128, lr=1e-3,
              device="cpu", labels=LABELS, target="cause"):
    """Train the TF-IDF MLP and save the checkpoint to ``out``; returns the training report.

    With ``val`` rows, the epoch with the lowest validation cross entropy is kept (stopping early
    after ``settings.patience`` epochs without improvement). With ``val=None`` (cross-validation
    already chose the epoch count), training runs exactly ``settings.epochs`` and keeps the last.
    """
    if min(settings.epochs, settings.batch_size, settings.max_features, hidden) < 1 or lr <= 0:
        raise ValueError("MLP training parameters must be positive")
    if val is None and settings.patience:
        raise ValueError("Early stopping requires validation rows")

    torch.manual_seed(settings.seed)
    vectorizer = TfidfVectorizer(max_features=settings.max_features, ngram_range=NGRAM_RANGE, dtype=np.float32)
    # Vocabulary and IDF come from training rows only, never from a holdout.
    x_train = torch.from_numpy(_dense(vectorizer, train, settings.use_checks, fit=True)).to(device)
    x_val = None if val is None else torch.from_numpy(_dense(vectorizer, val, settings.use_checks, fit=False)).to(device)
    y_train = _targets(train, labels, target, device)
    y_val = None if val is None else _targets(val, labels, target, device)

    model = MLP(x_train.shape[1], hidden, len(labels)).to(device)
    best_state, best_epoch, history = _fit(model, x_train, y_train, x_val, y_val, settings, lr, len(labels))

    out.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out / WEIGHTS_FILE)
    vocabulary = {key: int(value) for key, value in vectorizer.vocabulary_.items()}
    write_json(out / VECTORIZER_FILE, {"vocabulary": vocabulary, "idf": vectorizer.idf_.tolist(),
                                       "ngram_range": list(NGRAM_RANGE)})

    report = {"architecture": f"TF-IDF -> {hidden} ReLU -> {SECOND_LAYER} ReLU -> {len(labels)} {target} logits",
              "input_features": x_train.shape[1], "hidden": hidden,
              "parameters": sum(p.numel() for p in model.parameters()), "labels": list(labels), "target": target,
              "use_checks": settings.use_checks, "seed": settings.seed, "epochs": settings.epochs,
              "batch_size": settings.batch_size, "lr": lr,
              "best_epoch": best_epoch, "epochs_completed": len(history), "patience": settings.patience,
              "selection": ("minimum validation cross entropy" if val is not None
                            else "fixed epoch count selected by cross-validation"),
              "history": history}
    write_json(out / CONFIG_FILE, report)
    return report


def _fit(model, x_train, y_train, x_val, y_val, settings: TrainSettings, lr, n_classes):
    """The epoch loop; returns ``(best weights, best epoch, per-epoch history)``."""
    # Class weights make each cause count equally in the loss.
    weights = len(y_train) / (n_classes * torch.bincount(y_train, minlength=n_classes).clamp_min(1).float())
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(settings.seed)
    best_loss, best_state, best_epoch = float("inf"), None, 0
    history = []

    for epoch in range(1, settings.epochs + 1):
        train_loss = _train_one_epoch(model, x_train, y_train, weights, optimizer, generator, settings.batch_size)
        model.eval()

        if x_val is None:
            val_loss = None
            best_epoch, best_state = epoch, _snapshot(model)
        else:
            with torch.no_grad():
                val_loss = float(nn.functional.cross_entropy(model(x_val), y_val))
            if val_loss < best_loss:
                best_loss, best_epoch, best_state = val_loss, epoch, _snapshot(model)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if settings.verbose:
            shown = f"{val_loss:.4f}" if val_loss is not None else "n/a"
            print(f"MLP epoch {epoch}/{settings.epochs}: train_loss={train_loss:.4f}, val_loss={shown}", flush=True)
        if settings.patience and x_val is not None and epoch - best_epoch >= settings.patience:
            break

    return best_state, best_epoch, history


def _train_one_epoch(model, x_train, y_train, weights, optimizer, generator, batch_size) -> float:
    """One shuffled pass over the training rows; returns the mean training loss."""
    model.train()
    order = torch.randperm(len(y_train), generator=generator).tolist()
    total = 0.0
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(x_train[indices]), y_train[indices], weight=weights)
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * len(indices)
    return total / len(y_train)


def _texts(rows) -> list[str]:
    return [row["state"] for row in rows]


def _dense(vectorizer, rows, use_checks: bool, *, fit: bool) -> np.ndarray:
    """TF-IDF matrix (plus scaled comparison features when ``use_checks``) as a dense float32 array."""
    texts = _texts(rows)
    tfidf = (vectorizer.fit_transform(texts) if fit else vectorizer.transform(texts)).toarray().astype(np.float32)
    if not use_checks:
        return tfidf
    return np.hstack([tfidf, CHECK_SCALE * check_matrix(texts)])


def _targets(rows, labels, target, device):
    return torch.tensor([labels.index(row["expected"][target]) for row in rows], device=device)


def _snapshot(model) -> dict:
    return copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})


# ---------------------------------------------------------------- loading

def load_mlp(path: Path, device="cpu") -> MLPPredictor:
    config = read_json(path / CONFIG_FILE)
    features = read_json(path / VECTORIZER_FILE)

    vectorizer = TfidfVectorizer(vocabulary=features["vocabulary"], ngram_range=tuple(features["ngram_range"]),
                                 dtype=np.float32)
    vectorizer.idf_ = np.asarray(features["idf"], dtype=np.float32)

    model = MLP(config["input_features"], config["hidden"], len(config["labels"]))
    model.load_state_dict(torch.load(path / WEIGHTS_FILE, map_location="cpu", weights_only=True))
    return MLPPredictor(model, vectorizer, device, use_checks=config.get("use_checks", False))
