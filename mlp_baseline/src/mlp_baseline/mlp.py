from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn

from .data import LABELS, write_json


class MLP(nn.Module):
    def __init__(self, inputs: int, hidden: int = 128, classes: int = len(LABELS)):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(), nn.Dropout(0.1),
                                     nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, classes))

    def forward(self, values):
        return self.network(values)


class MLPPredictor:
    def __init__(self, model, vectorizer, device="cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.vectorizer = vectorizer

    @torch.no_grad()
    def predict_proba(self, rows):
        values = self.vectorizer.transform([row["state"] for row in rows]).toarray()
        return self.model(torch.from_numpy(values).to(self.device)).softmax(-1).cpu().numpy()

    def predict(self, rows):
        result = self.predict_proba(rows)
        # The three cause classes form a partition; correctness is P(cause=none).
        return result, result[:, 0]


def train_mlp(train, val, out: Path, *, seed=42, epochs=40, batch_size=16,
              max_features=4096, hidden=128, lr=1e-3, device="cpu",
              labels=LABELS, target="cause", patience=None, verbose=False):
    if min(epochs, batch_size, max_features, hidden) < 1 or lr <= 0:
        raise ValueError("MLP training parameters must be positive")
    torch.manual_seed(seed)
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), dtype=np.float32)
    # Learn vocabulary and IDF only from training, never from either holdout.
    x_train = torch.from_numpy(vectorizer.fit_transform([row["state"] for row in train]).toarray()).to(device)
    x_val = torch.from_numpy(vectorizer.transform([row["state"] for row in val]).toarray()).to(device)
    y_train = torch.tensor([labels.index(row["expected"][target]) for row in train], device=device)
    y_val = torch.tensor([labels.index(row["expected"][target]) for row in val], device=device)
    model = MLP(x_train.shape[1], hidden, len(labels)).to(device)
    weights = len(train) / (len(labels) * torch.bincount(y_train, minlength=len(labels)).clamp_min(1).float())
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    best_loss, best_state, best_epoch = float("inf"), None, 0
    history = []
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train), generator=generator).tolist()
        total = 0.0
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(x_train[indices]), y_train[indices], weight=weights)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        model.eval()
        with torch.no_grad():
            val_loss = float(nn.functional.cross_entropy(model(x_val), y_val))
        history.append({"epoch": epoch + 1, "train_loss": total / len(train), "val_loss": val_loss})
        if verbose:
            print(f"MLP epoch {epoch + 1}/{epochs}: train_loss={total / len(train):.4f}, val_loss={val_loss:.4f}", flush=True)
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch + 1
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        if patience and epoch + 1 - best_epoch >= patience:
            break
    out.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out / "model.pt")
    write_json(out / "vectorizer.json", {"vocabulary": {key: int(value) for key, value in vectorizer.vocabulary_.items()}, "idf": vectorizer.idf_.tolist(),
                                         "ngram_range": [1, 2]})
    report = {"architecture": f"TF-IDF -> {hidden} ReLU -> 64 ReLU -> {len(labels)} {target} logits",
              "input_features": x_train.shape[1], "hidden": hidden,
              "parameters": sum(p.numel() for p in model.parameters()), "labels": list(labels), "target": target,
              "seed": seed, "epochs": epochs, "batch_size": batch_size, "lr": lr,
              "best_epoch": best_epoch, "epochs_completed": len(history), "patience": patience,
              "selection": "minimum validation cross entropy", "history": history}
    write_json(out / "model_config.json", report)
    return report


def load_mlp(path: Path, device="cpu"):
    config = json.loads((path / "model_config.json").read_text())
    features = json.loads((path / "vectorizer.json").read_text())
    vectorizer = TfidfVectorizer(vocabulary=features["vocabulary"], ngram_range=tuple(features["ngram_range"]), dtype=np.float32)
    vectorizer.idf_ = np.asarray(features["idf"], dtype=np.float32)
    model = MLP(config["input_features"], config["hidden"], len(config["labels"]))
    model.load_state_dict(torch.load(path / "model.pt", map_location="cpu", weights_only=True))
    return MLPPredictor(model, vectorizer, device)
