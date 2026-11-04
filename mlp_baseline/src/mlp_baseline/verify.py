"""``mlp-baseline verify``: check the released files against RELEASE.json and the saved predictions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .common import read_jsonl
from .injection_model import load_injection_mlp, validate_injection_splits

INJECTION_DATA = Path("data/injection")
INJECTION_EXPERIMENT = Path("experiments/injection")
SEEDS = (42, 43, 44)


def verify_release(root: Path):
    """Every file hash in RELEASE.json matches, the injection splits are valid, and each exported
    checkpoint reproduces its saved test probabilities."""
    root = root.resolve()
    manifest = json.loads((root / "RELEASE.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        path = root / name
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError(f"Unsafe release path: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Release file hash mismatch: {name}")

    splits, _ = validate_injection_splits(root / INJECTION_DATA)
    checked = {}
    for seed in SEEDS:
        seed_dir = root / INJECTION_EXPERIMENT / f"seed{seed}"
        values = load_injection_mlp(seed_dir / "model").predict_proba(splits["test"])
        expected = read_jsonl(seed_dir / "predictions" / "test_predictions.jsonl")
        if len(expected) != len(values):
            raise ValueError("Saved test predictions have the wrong row count")
        np.testing.assert_allclose(values[:, 1], [row["injection_probability"] for row in expected],
                                   atol=1e-6, rtol=1e-6)
        checked[str(seed)] = len(values)
    return {"status": "verified", "files_checked": len(manifest["files_sha256"]),
            "exported_test_predictions_checked": checked, "split_groups_disjoint": True}
