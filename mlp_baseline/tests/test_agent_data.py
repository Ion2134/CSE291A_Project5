import json

import pytest

np = pytest.importorskip("numpy")
from mlp_baseline.agent_data import combine_agent_runs, export_laya_banking, validate_combined_dataset
from mlp_baseline.data import LABELS, input_id, request_group

QUESTIONS = {"output_correct": {"type": "noul"}, "cause": {"type": "choice"}}
# Group names avoid digits: request_group normalizes numbers to <number>.
NAMES = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf")


def _row(group, variant, cause, *, repair_case=False):
    if repair_case:
        state = (f"user_message: Handle the {group} request number {variant}.\n"
                 "trusted: {}\n"
                 "tool: list_accounts\n"
                 "model_call: {}\n"
                 "received_args: {}\n"
                 "output: {'accounts': [{'account_id': 'acct_synth_0001'}]}\n")
    else:
        state = (f"user_message: Handle the {group} request number {variant}.\n"
                 "model_call: lookup\n"
                 f"output: {cause}\n")
    return {"state": state, "questions": QUESTIONS,
            "expected": {"output_correct": cause == "none", "cause": cause}}


def _example(row, example_id, *, operator="benign", repair_case=False):
    inputs = {"state_text": row["state"], "trusted": {}, "model_call": {}, "received_args": {}, "output": {}}
    if repair_case:
        inputs["output"] = {"accounts": [{"account_id": "acct_synth_0001"}]}
    return {
        "example_id": example_id,
        "input": inputs,
        "label": {"output_correct": row["expected"]["output_correct"], "cause": row["expected"]["cause"]},
        "meta": {"operator": operator},
    }


def _write_run(runs_dir, name, rows, *, operators=None):
    run_dir = runs_dir / name
    run_dir.mkdir(parents=True)
    operators = operators or ["benign"] * len(rows)
    examples = [_example(row, f"{name}-{index}", operator=operators[index],
                         repair_case=operators[index] == "mcp_io_mismatch")
                for index, row in enumerate(rows)]
    (run_dir / "laya_train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (run_dir / "examples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in examples))
    return run_dir


def _group_rows(groups):
    rows, operators = [], []
    for group in groups:
        for variant, cause in enumerate(("none", "none", "agent", "adversary")):
            rows.append(_row(group, variant, cause))
            operators.append("benign")
    return rows, operators


def test_combine_merges_runs_repairs_dedups_and_preserves_sources(tmp_path):
    runs_dir = tmp_path / "runs"
    shared, shared_ops = _group_rows(NAMES[:6])
    repair_row = _row(NAMES[0], 99, "adversary", repair_case=True)
    run_a_rows = shared + [shared[0], repair_row]
    run_a_ops = shared_ops + ["benign", "mcp_io_mismatch"]
    extra, extra_ops = _group_rows([NAMES[6]])
    run_b_rows = shared[:8] + extra
    run_b_ops = shared_ops[:8] + extra_ops
    first = _write_run(runs_dir, "run_a", run_a_rows, operators=run_a_ops)
    second = _write_run(runs_dir, "run_b", run_b_rows, operators=run_b_ops)
    originals = {path: path.read_bytes() for path in runs_dir.rglob("*") if path.is_file()}

    out = tmp_path / "combined"
    manifest = combine_agent_runs(runs_dir, out, seed=11, runs=("run_a", "run_b"))

    assert manifest["raw_rows"] == len(run_a_rows) + len(run_b_rows)
    assert manifest["repaired_rows"] == 1
    correction = manifest["label_corrections"][0]
    assert correction["run"] == "run_a" and correction["after"] == {"output_correct": True, "cause": "none"}
    assert manifest["duplicate_rows_removed"] == len(run_a_rows) + len(run_b_rows) - manifest["unique_inputs"]
    assert manifest["label_counts"] == {"none": 15, "agent": 7, "adversary": 7}
    provenance = [json.loads(line) for line in (out / "provenance.jsonl").read_text().splitlines()]
    assert sum(len(entry["sources"]) for entry in provenance) == len(run_a_rows) + len(run_b_rows)
    cross_run = [entry for entry in provenance
                 if {source["run"] for source in entry["sources"]} == {"run_a", "run_b"}]
    assert len(cross_run) == 8
    assert {path: path.read_bytes() for path in runs_dir.rglob("*") if path.is_file()} == originals
    assert combine_agent_runs(runs_dir, tmp_path / "again", seed=11, runs=("run_a", "run_b")) == manifest

    splits, reloaded = validate_combined_dataset(out)
    assert reloaded == manifest
    assert {request_group(row) for row in splits["train"]}.isdisjoint(
        request_group(row) for row in splits["test"])
    assert all({row["expected"]["cause"] for row in split} == set(LABELS) for split in splits.values())
    with pytest.raises(ValueError, match="new or empty"):
        combine_agent_runs(runs_dir, out, seed=11, runs=("run_a", "run_b"))


def test_conflicting_labels_across_runs_fail_without_writing(tmp_path):
    runs_dir = tmp_path / "runs"
    rows, ops = _group_rows(NAMES[:6])
    conflict = {**rows[0], "expected": {"output_correct": False, "cause": "agent"}}
    _write_run(runs_dir, "run_a", rows, operators=ops)
    _write_run(runs_dir, "run_b", [conflict] + rows[1:], operators=ops)
    out = tmp_path / "combined"
    with pytest.raises(ValueError, match="Unresolved conflicting labels"):
        combine_agent_runs(runs_dir, out, seed=11, runs=("run_a", "run_b"))
    assert not out.exists()


def test_misaligned_run_files_fail(tmp_path):
    runs_dir = tmp_path / "runs"
    rows, ops = _group_rows(NAMES[:6])
    _write_run(runs_dir, "run_a", rows, operators=ops)
    run_dir = _write_run(runs_dir, "run_b", rows, operators=ops)
    lines = (run_dir / "examples.jsonl").read_text().splitlines()
    (run_dir / "examples.jsonl").write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError, match="different row counts"):
        combine_agent_runs(runs_dir, tmp_path / "combined", seed=11, runs=("run_a", "run_b"))


def test_export_laya_banking_freezes_test_and_carves_val(tmp_path):
    runs_dir = tmp_path / "runs"
    rows, ops = _group_rows(NAMES)
    _write_run(runs_dir, "run_a", rows, operators=ops)
    _write_run(runs_dir, "run_b", rows, operators=ops)
    combined = tmp_path / "combined"
    combine_agent_runs(runs_dir, combined, seed=11, runs=("run_a", "run_b"))
    source = json.loads((combined / "manifest.json").read_text())
    out = tmp_path / "laya"
    manifest = export_laya_banking(combined, out, seed=11, val_fraction=0.25)

    assert set(manifest["splits"]) == {"train", "val", "test"}
    assert manifest["splits"]["test"]["input_ids"] == source["splits"]["test"]["input_ids"]
    assert manifest["source_split_sha256"] == source["split_sha256"]
    rows_by_split = {
        name: [json.loads(line) for line in (out / f"{name}.jsonl").read_text().splitlines()]
        for name in ("train", "val", "test")
    }
    for name, rows in rows_by_split.items():
        assert rows and all(set(row) == {"state", "questions", "expected"} for row in rows)
        assert {row["expected"]["cause"] for row in rows} == set(LABELS)
    ids = {name: {input_id(row) for row in part} for name, part in rows_by_split.items()}
    assert not ids["train"] & ids["val"] and not ids["train"] & ids["test"] and not ids["val"] & ids["test"]
    assert {request_group(row) for row in rows_by_split["train"]}.isdisjoint(
        request_group(row) for row in rows_by_split["val"])
    train_val_groups = {request_group(row) for row in rows_by_split["train"] + rows_by_split["val"]}
    test_groups = {request_group(row) for row in rows_by_split["test"]}
    assert train_val_groups.isdisjoint(test_groups)
    with pytest.raises(ValueError, match="new or empty"):
        export_laya_banking(combined, out, seed=11)
