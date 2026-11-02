import json
import pytest
np = pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("sklearn")
from mlp_baseline.injection_data import INJECTION_QUESTIONS
from mlp_baseline.data import LABELS, input_id, read_rows, request_group, split_rows, write_jsonl
from mlp_baseline.metrics import evaluate, fit_temperature
from mlp_baseline.mlp import train_mlp, load_mlp
BANK_QUESTIONS = {"output_correct": {"type": "noul"}, "cause": {"type": "choice", "criteria": {key: key for key in LABELS}}}


@pytest.fixture
def rows():
    result = []
    for name in ("oak", "elm", "ash", "birch", "pine", "fir", "maple", "cedar", "willow", "beech"):
        for cause in LABELS:
            result.append({
                "state": f"user_message: Handle the {name} request.\nmodel_call: lookup\noutput: {cause}\n",
                "questions": BANK_QUESTIONS,
                "expected": {"output_correct": cause == "none", "cause": cause},
            })
    return result


def test_splits_are_deterministic_and_keep_requests_together(rows):
    splits, manifest = split_rows(rows, seed=11)
    assert split_rows(rows, seed=11)[1] == manifest
    assert split_rows(rows, seed=12)[1] != manifest
    assert sum(map(len, splits.values())) == len(rows)
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert {request_group(row) for row in splits[a]}.isdisjoint(request_group(row) for row in splits[b])
        assert {input_id(row) for row in splits[a]}.isdisjoint(input_id(row) for row in splits[b])
    assert set(manifest["splits"]["train"]["label_counts"]) == set(LABELS)


def test_random_payment_ids_do_not_create_independent_cases(tmp_path, rows):
    first = {**rows[0], "state": rows[0]["state"] + "payment_id: pay_synth_1234abcd\n"}
    second = {**first, "state": first["state"].replace("1234abcd", "fedc9876")}
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [first, second])
    unique, removed = read_rows(path)
    assert unique == [first]
    assert removed == 1
    write_jsonl(path, [first, {**second, "expected": {"output_correct": False, "cause": "agent"}}])
    with pytest.raises(ValueError, match="Conflicting labels"):
        read_rows(path)


def test_metrics_perfect_predictions_and_single_class(rows):
    targets = [LABELS.index(row["expected"]["cause"]) for row in rows]
    correct = [row["expected"]["output_correct"] for row in rows]
    result = evaluate(rows, np.eye(3)[targets], correct)
    assert result["cause_macro_f1"] == result["error_auroc"] == 1
    assert result["false_alarm_rate"] == result["missed_error_rate"] == 0
    assert result["correctness_brier"] == 0
    one_class = evaluate(rows[:1], [[1, 0, 0]], [1])
    assert one_class["error_auroc"] is None
    assert one_class["missed_error_rate"] is None
    json.dumps(one_class, allow_nan=False)


def test_metrics_count_false_alarms_and_missed_errors(rows):
    result = evaluate(rows[:3], [[0, 1, 0], [1, 0, 0], [0, 0, 1]], [0.1, 0.9, 0.1])
    assert result["false_alarm_rate"] == 1
    assert result["missed_error_rate"] == 0.5
    assert result["cause_confusion_matrix"] == [[0, 1, 0], [1, 0, 0], [0, 0, 1]]


def test_metrics_error_type_attribution(rows):
    targets = [LABELS.index(row["expected"]["cause"]) for row in rows]
    correct = [row["expected"]["output_correct"] for row in rows]
    perfect = evaluate(rows, np.eye(3)[targets], correct)
    assert perfect["error_type_accuracy"] == 1
    assert perfect["error_type_support"] == 20
    assert perfect["error_type_accuracy_when_detected"] == 1
    assert perfect["error_type_detected_support"] == 20
    # Predicting "none" everywhere: attribution is zero and nothing is detected.
    missed = evaluate(rows, np.tile([1.0, 0, 0], (len(rows), 1)), [1.0] * len(rows))
    assert missed["error_type_accuracy"] == 0
    assert missed["error_type_accuracy_when_detected"] is None
    assert missed["error_type_detected_support"] == 0
    json.dumps(missed, allow_nan=False)
    # Swapped agent/adversary: errors are detected but attributed to the wrong type.
    swap = {"agent": "adversary", "adversary": "agent"}
    swapped = np.eye(3)[[LABELS.index(swap.get(row["expected"]["cause"], "none")) for row in rows]]
    confused = evaluate(rows, swapped, correct)
    assert confused["error_type_accuracy"] == 0
    assert confused["error_type_accuracy_when_detected"] == 0
    assert confused["error_type_detected_support"] == 20
    # No error rows at all: both metrics are guarded.
    correct_only = [row for row in rows if row["expected"]["cause"] == "none"]
    clean = evaluate(correct_only, np.eye(3)[[0] * len(correct_only)], [1.0] * len(correct_only))
    assert clean["error_type_accuracy"] is None
    assert clean["error_type_support"] == 0
    assert clean["error_type_accuracy_when_detected"] is None
    json.dumps(clean, allow_nan=False)


def test_small_validation_does_not_claim_calibration():
    calibration = fit_temperature([[0.2, 0.8]], [1])
    assert calibration["fitted"] is False
    assert calibration["temperature"] == 1


def test_temperature_reduces_overconfidence_on_validation():
    from mlp_baseline.metrics import scale_temperature
    values = np.tile([0.99, 0.01], (40, 1))
    targets = np.array([0, 1] * 20)
    calibration = fit_temperature(values, targets)
    assert calibration["fitted"] is True
    assert calibration["temperature"] > 1
    calibrated = scale_temperature(values, calibration["temperature"])
    before = -np.log(values[np.arange(len(targets)), targets]).mean()
    after = -np.log(calibrated[np.arange(len(targets)), targets]).mean()
    assert after < before


def test_account_and_amount_variants_stay_in_one_request_group(rows):
    first = {**rows[0], "state": "user_message: Pay payee_synth_0007 $25 from acct_synth_0031.\n"}
    second = {**first, "state": "user_message: Pay payee_synth_0099 $100 from acct_synth_0042.\n"}
    assert request_group(first) == request_group(second)
    assert input_id(first) != input_id(second)


def test_mlp_vocabulary_does_not_learn_from_validation(tmp_path, rows):
    splits, _ = split_rows(rows)
    validation = [{**row, "state": row["state"] + " neverseenvalidationtoken"} for row in splits["val"]]
    train_mlp(splits["train"], validation, tmp_path, epochs=2, max_features=64)
    predictor = load_mlp(tmp_path)
    assert "neverseenvalidationtoken" not in predictor.vectorizer.vocabulary_
    before = predictor.predict(splits["test"])
    after = load_mlp(tmp_path).predict(splits["test"])
    for x, y in zip(before, after):
        np.testing.assert_array_equal(x, y)
