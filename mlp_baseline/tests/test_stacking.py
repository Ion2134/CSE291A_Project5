import hashlib
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("sklearn")
from mlp_baseline.data import LABELS
from mlp_baseline.stacking import (StackedMonitor, fit_stacker, load_stacker, save_stacker,
                                   stacker_features)

QUESTIONS = {"output_correct": {"type": "noul"}, "cause": {"type": "choice"}}
REAL_CHECKPOINT = Path(__file__).resolve().parents[1] / "models/injection/seed42"


def _row(variant, cause):
    return {
        "state": (f"user_message: Handle the shared request number {variant}.\n"
                  f"model_call: lookup\noutput: {cause} token{cause}\n"),
        "questions": QUESTIONS,
        "expected": {"output_correct": cause == "none", "cause": cause},
    }


def _rows(per_class=6):
    return [_row(variant, cause) for cause in LABELS for variant in range(per_class)]


class FakeFrozen:
    def predict_proba(self, rows):
        return np.tile([0.7, 0.3], (len(rows), 1))


def test_stacker_features_include_injection_probability():
    rows = _rows(per_class=2)
    frozen = FakeFrozen()
    vectorizer, _ = fit_stacker(_rows(), frozen)
    features = stacker_features(rows, frozen, vectorizer)
    assert features.shape == (len(rows), 1 + len(vectorizer.vocabulary_))
    np.testing.assert_allclose(features[:, 0], 0.3)


def test_stacker_vocabulary_learns_from_first_party_train_only():
    frozen = FakeFrozen()
    train = _rows()
    held_out = [{**_row(0, "none"), "state": _row(0, "none")["state"] + " neverseenheldouttoken"}]
    vectorizer, stacker = fit_stacker(train, frozen)
    assert "neverseenheldouttoken" not in vectorizer.vocabulary_
    assert set(stacker.classes_) == {0, 1, 2}
    with pytest.raises(ValueError, match="every cause class"):
        fit_stacker([row for row in train if row["expected"]["cause"] != "agent"], frozen)


def test_stacked_monitor_probabilities_form_a_partition():
    monitor = StackedMonitor(FakeFrozen(), *fit_stacker(_rows(), FakeFrozen()),
                             {"cause": {"temperature": 1.0}, "correctness": {"temperature": 1.0}})
    rows = _rows(per_class=2)
    cause, correct = monitor.predict(rows)
    assert cause.shape == (len(rows), len(LABELS)) and correct.shape == (len(rows),)
    np.testing.assert_allclose(cause.sum(axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(correct, cause[:, 0], atol=1e-6)


@pytest.mark.skipif(not REAL_CHECKPOINT.exists(), reason="release injection checkpoint is missing")
def test_save_load_roundtrip_preserves_predictions_and_frozen_base(tmp_path):
    from mlp_baseline.injection_mlp import load_injection_mlp
    frozen_bytes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in REAL_CHECKPOINT.iterdir()}
    frozen = load_injection_mlp(REAL_CHECKPOINT)
    vectorizer, stacker = fit_stacker(_rows(), frozen)
    calibration = {"cause": {"temperature": 1.0}, "correctness": {"temperature": 1.0}, "source": "test"}
    monitor = StackedMonitor(frozen, vectorizer, stacker, calibration)
    rows = _rows(per_class=2)
    before = monitor.predict(rows)
    save_stacker(tmp_path / "stacker", vectorizer, stacker, calibration,
                 base_checkpoint=str(REAL_CHECKPOINT), training={"model": "test"})
    restored = load_stacker(tmp_path / "stacker")
    after = restored.predict(rows)
    np.testing.assert_allclose(before[0], after[0], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(before[1], after[1], atol=1e-6, rtol=1e-6)
    assert {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in REAL_CHECKPOINT.iterdir()} == frozen_bytes
