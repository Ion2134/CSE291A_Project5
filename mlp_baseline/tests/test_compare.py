import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("sklearn")
from mlp_baseline.agent_data import combine_agent_runs
from mlp_baseline.compare import run_agent_comparison
from mlp_baseline.data import LABELS

QUESTIONS = {"output_correct": {"type": "noul"}, "cause": {"type": "choice"}}
REAL_CHECKPOINT = Path(__file__).resolve().parents[1] / "models/injection/seed42"
NAMES = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot")


def _row(group, variant, cause):
    return {
        "state": (f"user_message: Handle the {group} request number {variant}.\n"
                  f"model_call: lookup\noutput: {cause} {cause}token\n"),
        "questions": QUESTIONS,
        "expected": {"output_correct": cause == "none", "cause": cause},
    }


def _example(row, example_id):
    return {
        "example_id": example_id,
        "input": {"state_text": row["state"], "trusted": {}, "model_call": {},
                  "received_args": {}, "output": {}},
        "label": {"output_correct": row["expected"]["output_correct"], "cause": row["expected"]["cause"]},
        "meta": {"operator": "benign"},
    }


def _write_run(runs_dir, name, rows):
    run_dir = runs_dir / name
    run_dir.mkdir(parents=True)
    (run_dir / "laya_train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (run_dir / "examples.jsonl").write_text(
        "".join(json.dumps(_example(row, f"{name}-{index}")) + "\n" for index, row in enumerate(rows)))


@pytest.mark.skipif(not REAL_CHECKPOINT.exists(), reason="release injection checkpoint is missing")
def test_run_agent_comparison_end_to_end(tmp_path):
    runs_dir = tmp_path / "runs"
    rows = [_row(group, variant, cause) for group in NAMES
            for variant, cause in enumerate(("none", "none", "agent", "adversary", "none", "adversary"))]
    _write_run(runs_dir, "run_a", rows)
    _write_run(runs_dir, "run_b", rows[:12])
    data = tmp_path / "combined"
    combine_agent_runs(runs_dir, data, seed=3, runs=("run_a", "run_b"))
    out = tmp_path / "report"
    result = run_agent_comparison(data, out, seed=3, k=2, epochs=3, batch_size=8,
                                  max_features=256, injection_checkpoint=REAL_CHECKPOINT)

    assert result["status"] == "complete"
    assert result["cv"]["k"] == 2
    scenarios = result["scenarios"]
    assert set(scenarios) == {"scenario_a_first_party_mlp", "scenario_b_hybrid_stacking"}
    test_rows = [json.loads(line) for line in (data / "test.jsonl").read_text().splitlines()]
    for name, scenario in scenarios.items():
        assert scenario["test"]["samples"] == len(test_rows)
        assert scenario["test"]["cause_labels"] == list(LABELS)
        assert scenario["calibration"]["source"] == "out-of-fold train-pool predictions only"
        assert set(scenario["test_by_source_run"]) <= {"run_a", "run_b"}
    for key, values in result["head_to_head"].items():
        assert set(values) == {"scenario_a", "scenario_b", "majority_baseline"}
    predictions = [json.loads(line) for line in (out / "test_predictions.jsonl").read_text().splitlines()]
    assert len(predictions) == len(test_rows)
    assert all(set(row["scenario_a"]["cause_probabilities"]) == set(LABELS) for row in predictions)
    # Saved artifacts reload and reproduce the reported predictions.
    from mlp_baseline.stacking import load_stacker
    restored = load_stacker(out / "models/scenario-b-hybrid-stacking/final")
    cause, _ = restored.predict(test_rows)
    import numpy as np
    np.testing.assert_allclose(
        cause, [list(row["scenario_b"]["cause_probabilities"].values()) for row in predictions],
        atol=1e-6, rtol=1e-6)
    with pytest.raises(ValueError, match="new or empty"):
        run_agent_comparison(data, out, seed=3, k=2, epochs=2)
