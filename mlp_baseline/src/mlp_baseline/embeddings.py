"""Sentence embeddings of monitor states, from a locally cached Hugging Face model (never downloaded here).

Embeddings are cached on disk by (model, text) hash, so a sweep embeds each distinct state once.
Set ``MLP_EMBED_CACHE`` to move the cache.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

DEFAULT_EMBEDDING = "BAAI/bge-small-en-v1.5"
MAX_TOKENS = 512
BATCH = 16
# Most informative lines first, so truncation at MAX_TOKENS drops the least useful text (the user's wording).
_ORDER = ("trusted", "tool", "model_call", "received_args", "output", "prior_calls", "call_index", "user_message")


def embed_text(state: str) -> str:
    """The state with its lines regrouped in ``_ORDER`` (continuation lines stay with their section)."""
    sections, current = {}, None
    for line in state.splitlines():
        key = line.split(":", 1)[0]
        if not line.startswith(" ") and key in _ORDER:
            current = key
        sections.setdefault(current or "user_message", []).append(line)
    return "\n".join(line for key in _ORDER for line in sections.get(key, []))


def _cache_path(name: str) -> Path:
    root = Path(os.environ.get("MLP_EMBED_CACHE", Path.home() / ".cache" / "mlp_baseline_embeddings"))
    return root / (name.replace("/", "__") + ".npz")


def _key(name: str, text: str) -> str:
    return hashlib.sha256(f"{name}\x00{text}".encode()).hexdigest()


def embed_texts(states, name: str = DEFAULT_EMBEDDING) -> np.ndarray:
    """L2-normalised CLS embeddings, one row per state string."""
    texts = [embed_text(state) for state in states]
    keys = [_key(name, text) for text in texts]

    path = _cache_path(name)
    cache = {}
    if path.exists():
        with np.load(path) as stored:
            cache = dict(zip(stored["keys"].tolist(), stored["vectors"]))

    missing = {key: text for key, text in zip(keys, texts) if key not in cache}
    if missing:
        for key, vector in zip(missing, _encode(list(missing.values()), name)):
            cache[key] = vector
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npz")
        np.savez(temporary, keys=np.array(list(cache)), vectors=np.stack(list(cache.values())))
        os.replace(temporary, path)
    return np.stack([cache[key] for key in keys]).astype(np.float32)


def _encode(texts: list[str], name: str) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
    model = AutoModel.from_pretrained(name, local_files_only=True).eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(texts), BATCH):
            batch = tokenizer(texts[start:start + BATCH], padding=True, truncation=True, max_length=MAX_TOKENS,
                              return_tensors="pt")
            cls = model(**batch).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(cls, dim=-1).numpy())
    return np.concatenate(out)
