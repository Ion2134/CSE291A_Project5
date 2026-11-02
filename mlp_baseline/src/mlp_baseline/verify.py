from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .injection_mlp import load_injection_mlp, validate_injection_splits


def verify_release(root: Path):
    root = root.resolve()
    manifest = json.loads((root / "RELEASE.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        path = root / name
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError(f"Unsafe release path: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Release file hash mismatch: {name}")
    splits, _ = validate_injection_splits(root / "data/injection-v1")
    checked = {}
    for seed in (42, 43, 44):
        predictor = load_injection_mlp(root / f"models/injection/seed{seed}")
        values = predictor.predict_proba(splits["test"])
        expected = [json.loads(line) for line in (root / f"predictions/seed{seed}/test_predictions.jsonl").read_text().splitlines()]
        if len(expected) != len(values):
            raise ValueError("Saved test predictions have the wrong row count")
        np.testing.assert_allclose(values[:, 1], [row["injection_probability"] for row in expected], atol=1e-6, rtol=1e-6)
        checked[str(seed)] = len(values)
    return {"status": "verified", "files_checked": len(manifest["files_sha256"]),
            "exported_test_predictions_checked": checked, "split_groups_disjoint": True}
