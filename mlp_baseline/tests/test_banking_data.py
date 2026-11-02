from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from mlp_baseline.banking_data import prepare_training_data


def _write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_prepare_real_triton_run_repairs_conflicts_and_preserves_sources(tmp_path):
    source = Path(__file__).resolve().parents[1] / "data/banking/triton42-v2/raw"
    for name in ("laya_train.jsonl", "examples.jsonl", "baseline.json"):
        shutil.copyfile(source / name, tmp_path / name)
    originals = {path: path.read_bytes() for path in tmp_path.iterdir()}
    report = prepare_training_data(tmp_path)
    prepared = tmp_path / "prepared"
    cleaned = [json.loads(line) for line in (prepared / "laya_train.jsonl").read_text().splitlines()]
    raw = [json.loads(line) for line in (tmp_path / "laya_train.jsonl").read_text().splitlines()]
    provenance = [json.loads(line) for line in (prepared / "provenance.jsonl").read_text().splitlines()]
    assert report["repaired_rows"] == 2
    assert report["duplicate_rows_removed"] == 27
    assert report["unique_inputs"] == len(cleaned) == 58
    assert report["label_counts"] == {"none": 45, "adversary": 10, "agent": 3}
    assert len({json.dumps({"state": row["state"], "questions": row["questions"]}, sort_keys=True) for row in cleaned}) == len(cleaned)
    assert report["remaining_conflicting_inputs"] == 0
    assert {change["source_line"] for change in report["label_corrections"]} == {32, 68}
    # Every original row has a traceable destination; genuine error labels survive.
    assert sum(len(group["sources"]) for group in provenance) == len(raw)
    corrected = {change["source_line"] for change in report["label_corrections"]}
    for group in provenance:
        row = cleaned[group["clean_line"] - 1]
        for entry in group["sources"]:
            before = raw[entry["source_line"] - 1]
            assert row["state"] == before["state"]
            assert row["questions"] == before["questions"]
            if entry["source_line"] not in corrected:
                assert row["expected"] == before["expected"]
    assert all(path.read_bytes() == content for path, content in originals.items())
    first_output = {path.name: path.read_bytes() for path in prepared.iterdir()}
    assert prepare_training_data(tmp_path) == report
    assert {path.name: path.read_bytes() for path in prepared.iterdir()} == first_output


def test_unknown_conflicting_labels_are_rejected_without_overwriting_output(tmp_path):
    row = {"state": "same input", "questions": {}, "expected": {"output_correct": True, "cause": "none"}}
    _write_rows(tmp_path / "laya_train.jsonl", [
        row, {**row, "expected": {"output_correct": False, "cause": "agent"}},
    ])
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    sentinel = prepared / "laya_train.jsonl"
    sentinel.write_text("previous output\n")
    with pytest.raises(ValueError, match="Unresolved conflicting labels at rows 1 and 2"):
        prepare_training_data(tmp_path)
    assert sentinel.read_text() == "previous output\n"
    assert not (prepared / "data_report.json").exists()


def test_different_question_schemas_are_distinct_inputs(tmp_path):
    row = {"state": "same state", "questions": {"cause": {"type": "choice"}}, "expected": {"cause": "none"}}
    _write_rows(tmp_path / "laya_train.jsonl", [row, row, {**row, "questions": {"other": {"type": "noul"}}}])
    report = prepare_training_data(tmp_path)
    assert report["unique_inputs"] == 2
    assert report["duplicate_rows_removed"] == 1


def test_misaligned_source_files_are_rejected(tmp_path):
    _write_rows(tmp_path / "laya_train.jsonl", [{"state": "one", "questions": {}, "expected": {"cause": "none"}}])
    _write_rows(tmp_path / "examples.jsonl", [{"input": {"state_text": "other"}, "label": {"output_correct": True, "cause": "none"}}])
    with pytest.raises(ValueError, match="Source files disagree"):
        prepare_training_data(tmp_path)
    assert not (tmp_path / "prepared").exists()


def test_raw_run_cannot_be_output_directory(tmp_path):
    with pytest.raises(ValueError, match="separate output directory"):
        prepare_training_data(tmp_path, tmp_path)
