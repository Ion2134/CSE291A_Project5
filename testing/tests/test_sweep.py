"""Tests for the unified tester. Model training is replaced by tests/fake_model.py."""
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("sklearn")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sweep_helpers import models  # noqa: E402
from sweep_helpers.config import Sweep, Unit  # noqa: E402
from sweep_helpers.models import build_command, run_models  # noqa: E402
from sweep_helpers.splits import make_splits  # noqa: E402
from sweep_helpers.summary import summarize  # noqa: E402

FAKE_MODEL = Path(__file__).with_name("fake_model.py")
QUESTIONS = {"output_correct": {"type": "noul"}, "cause": {"type": "choice"}}
# Group names avoid digits: request grouping normalizes numbers.
GROUPS = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet")


def _write_run(runs_dir: Path, name: str, rows: list[dict]) -> None:
    run_dir = runs_dir / name
    run_dir.mkdir(parents=True)
    examples = [{"example_id": f"{name}-{index}",
                 "input": {"state_text": row["state"], "trusted": {}, "model_call": {},
                           "received_args": {}, "output": {}},
                 "label": dict(row["expected"]), "meta": {"operator": "benign"}}
                for index, row in enumerate(rows)]
    (run_dir / "laya_train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (run_dir / "examples.jsonl").write_text("".join(json.dumps(example) + "\n" for example in examples))


def _rows(groups):
    return [{"state": f"user_message: Handle the {group} request.\nmodel_call: lookup\noutput: {cause} case\n",
             "questions": QUESTIONS, "expected": {"output_correct": cause == "none", "cause": cause}}
            for group in groups for cause in ("none", "agent", "adversary")]


@pytest.fixture
def sweep(tmp_path):
    rows = _rows(GROUPS)
    _write_run(tmp_path / "runs", "run_a", rows[:18])
    _write_run(tmp_path / "runs", "run_b", rows[12:])  # Overlaps run_a: duplicates must merge.
    model = {"python": sys.executable, "args": []}
    config = {"name": "unit", "runs_dir": "runs", "runs": ["run_a", "run_b"],
              "split": {"seeds": [1, 2, 3], "k": 5, "max_test_fraction": 0.35, "laya_val_fraction": 0.2},
              "training_seeds": [7, 8],
              "models": {"mlp": dict(model), "laya": {**model, "training_seeds": [7]}}}
    return Sweep(config, repo=tmp_path)


@pytest.fixture
def fake_commands(monkeypatch):
    def fake_build_command(sweep, unit, native):
        test = sweep.fold_files(unit.split_seed, unit.fold).test
        style = "laya" if unit.model == "laya" else "mlp"
        return [sys.executable, str(FAKE_MODEL), "--format", style, "--test", str(test), "--out", str(native)]
    monkeypatch.setattr(models, "build_command", fake_build_command)


def test_splits_cover_every_group_once_and_keep_every_cause(sweep):
    manifest = make_splits(sweep)
    assert manifest["pool"]["rows"] == len(GROUPS) * 3
    assignments = set()
    for split in manifest["splits"]:
        tested = [group for fold in split["folds"] for group in fold["test_groups"]]
        assert sorted(tested) == sorted(manifest["pool"]["group_sizes"])  # Each group tested exactly once.
        assignments.add(frozenset(frozenset(fold["test_groups"]) for fold in split["folds"]))
        for fold in split["folds"]:
            assert not set(fold["laya_val_groups"]) & set(fold["test_groups"])
            assert set(fold["test_labels"]) == set(fold["train_labels"]) == {"none", "agent", "adversary"}
            assert fold["test_rows"] <= 0.35 * manifest["pool"]["rows"]
            files = sweep.fold_files(split["split_seed"], fold["fold"])
            ids = {name: {json.loads(line)["meta"]["input_id"] for line in getattr(files, name).open()}
                   for name in ("train", "val", "test")}
            assert not ids["train"] & ids["val"] and not ids["train"] & ids["test"] and not ids["val"] & ids["test"]
    assert len(assignments) == len(sweep.config["split"]["seeds"])  # Split seeds give distinct assignments.


def test_splits_refuse_changed_data_settings(sweep):
    make_splits(sweep)
    assert make_splits(sweep)["data_fingerprint"] == sweep.data_fingerprint()
    changed = Sweep({**sweep.config, "runs": ["run_a"]}, repo=sweep.repo)
    with pytest.raises(ValueError, match="different runs or split settings"):
        make_splits(changed)


def test_run_resume_and_summary(sweep, fake_commands, capsys):
    manifest = make_splits(sweep)
    run_models(sweep, ["mlp", "laya"], manifest)
    runs = sweep.out / "runs"
    assert len(list((runs / "mlp").glob("*/metrics.json"))) == 3 * 5 * 2
    assert len(list((runs / "laya").glob("*/metrics.json"))) == 3 * 5 * 1
    assert not list((runs / "laya").glob("*/native/model"))  # Checkpoints removed after scoring.

    capsys.readouterr()
    run_models(sweep, ["mlp"], manifest, split_seeds=[1], folds=[0])
    assert "0 trained, 2 already done" in capsys.readouterr().out

    summary = summarize(sweep, manifest)
    assert summary["models"]["mlp"]["repeats"] == 3 * 2
    assert summary["models"]["laya"]["repeats"] == 3
    assert summary["models"]["majority"]["repeats"] == 3
    # The fake model always puts most probability on the true cause.
    assert summary["models"]["mlp"]["metrics"]["cause_macro_f1"]["median"] == 1.0
    pairs = {(row["first"], row["second"], row["metric"]): row for row in summary["paired"]}
    assert pairs[("mlp", "laya", "cause_macro_f1")]["pairs"] == 3  # Only Laya's training seed matches.
    assert pairs[("majority", "mlp", "cause_macro_f1")]["first_better_rate"] == 0.0
    for name in ("results.csv", "summary.json", "SUMMARY.md",
                 "raw/repeats.csv", "raw/per_fold.csv", "raw/paired.csv"):
        assert (sweep.out / "summary" / name).exists()


def test_changed_model_settings_are_refused(sweep, fake_commands):
    manifest = make_splits(sweep)
    run_models(sweep, ["mlp"], manifest, split_seeds=[1], folds=[0], training_seeds=[7])
    models_config = {**sweep.config["models"], "mlp": {**sweep.config["models"]["mlp"], "args": ["--epochs", "3"]}}
    changed = Sweep({**sweep.config, "models": models_config}, repo=sweep.repo)
    with pytest.raises(ValueError, match="settings changed"):
        run_models(changed, ["mlp"], manifest, split_seeds=[1], folds=[0], training_seeds=[7])


def test_build_command_uses_each_folders_own_entry_point(sweep, tmp_path):
    native = tmp_path / "native"
    mlp = build_command(sweep, Unit("mlp", 1, 0, 7), native)
    assert mlp[1:5] == ["-m", "mlp_baseline.cli", "fit-predict", "--model"]
    laya = build_command(sweep, Unit("laya", 1, 0, 7), native)
    assert laya[1:4] == ["laya_finetuning/train_laya.py", "--task", "banking"]
    assert "--validation" in laya and "--output" in laya
