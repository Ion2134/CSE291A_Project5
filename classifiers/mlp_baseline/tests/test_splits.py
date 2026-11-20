import pytest

np = pytest.importorskip("numpy")
from mlp_baseline.splits import group_kfold, stratified_group_partition
from mlp_baseline.common import LABELS, request_group

QUESTIONS = {"output_correct": {"type": "noul"}, "cause": {"type": "choice"}}
# Group names avoid digits: request_group normalizes numbers to <number>.
NAMES = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet")


def _row(group, variant, cause):
    return {
        "state": (f"user_message: Handle the {group} request number {variant}.\n"
                  f"model_call: lookup\noutput: {cause}\n"),
        "questions": QUESTIONS,
        "expected": {"output_correct": cause == "none", "cause": cause},
    }


def _rows(agent_groups=None):
    agent_groups = set(range(10)) if agent_groups is None else set(agent_groups)
    rows = []
    for index in range(10):
        causes = ["none", "none", "adversary"]
        if index in agent_groups:
            causes.append("agent")
        rows.extend(_row(NAMES[index], variant, cause) for variant, cause in enumerate(causes))
    return rows


def test_partition_is_group_disjoint_stratified_and_deterministic():
    rows = _rows()
    parts, assignment = stratified_group_partition(rows, {"train": 6, "test": 4}, seed=11)
    assert set(assignment["train"]).isdisjoint(assignment["test"])
    assert sum(map(len, parts.values())) == len(rows)
    for part in parts.values():
        assert {row["expected"]["cause"] for row in part} == set(LABELS)
    again = stratified_group_partition(rows, {"train": 6, "test": 4}, seed=11)[1]
    assert again == assignment


def test_partition_requires_every_class_in_every_part():
    rows = _rows(agent_groups={0})
    with pytest.raises(ValueError, match="every class in every part"):
        stratified_group_partition(rows, {"a": 3, "b": 3, "c": 4}, seed=3)


def test_partition_rejects_bad_sizes():
    rows = _rows()
    with pytest.raises(ValueError, match="request group"):
        stratified_group_partition(rows, {"only": 10}, seed=3)
    with pytest.raises(ValueError, match="request group"):
        stratified_group_partition(rows, {"a": 5, "b": 6}, seed=3)


def test_kfold_degrades_until_every_fold_has_every_class():
    rows = _rows(agent_groups={0, 1})
    result = group_kfold(rows, k=5, seed=7)
    assert result["requested_k"] == 5
    assert result["k"] == 2
    assert result["degraded"] and "k=5" in result["degradation_reason"]
    for fold in result["folds"]:
        assert {row["expected"]["cause"] for row in fold["val"]} == set(LABELS)
        assert {request_group(row) for row in fold["train"]}.isdisjoint(
            request_group(row) for row in fold["val"])


def test_kfold_keeps_requested_k_when_data_allows_and_covers_rows_once():
    rows = _rows()
    result = group_kfold(rows, k=5, seed=7)
    assert result["k"] == 5 and not result["degraded"] and result["degradation_reason"] is None
    held_out = [row for fold in result["folds"] for row in fold["val"]]
    assert len(held_out) == len(rows)
    assert len({id(row) for row in held_out}) == len(rows)
    assert group_kfold(rows, k=5, seed=7)["folds"] == result["folds"]


def test_kfold_rejects_tiny_or_impossible_data():
    one_group = [_row("only", variant, cause) for variant, cause in enumerate(("none", "agent", "adversary"))]
    with pytest.raises(ValueError, match="at least two request groups"):
        group_kfold(one_group, k=2)
    with pytest.raises(ValueError, match="at least 2"):
        group_kfold(_rows(), k=1)
    with pytest.raises(ValueError, match="each contain every class"):
        group_kfold(_rows(agent_groups={0}), k=2)
