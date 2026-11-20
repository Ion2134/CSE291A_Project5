import json

import numpy as np
import pytest

from recovery_classifiers import CLASSIFIERS, Panel, decide, top_cause
from mlp_baseline.checks import rule_cause

OK = {"from_account_id": "a1", "payee_id": "p1", "amount_cents": 2500}
WRONG = {**OK, "amount_cents": 750}


def row(call, cause, group, received=None, output=None):
    state = (f"user_message: pay {group}\ntrusted: {{'from_account_id': 'a1', 'payee_id': 'p1', 'amount_cents': 2500}}\n"
             f"prior_calls:\ncall_index: 0\ntool: initiate_transfer\nmodel_call: {call}\n"
             f"received_args: {received or call}\noutput: {output or received or call}\n")
    return {"state": state, "expected": {"cause": cause, "output_correct": cause == "none"}, "questions": []}


TRAIN = ([row(OK, "none", i) for i in range(6)] + [row(WRONG, "agent", i) for i in range(6, 12)]
         + [row(OK, "adversary", i, received={**OK, "payee_id": "p9"}) for i in range(12, 16)])
TEST = [row(OK, "none", 20), row(WRONG, "agent", 21), row(OK, "adversary", 22, output={**OK, "_note": "x"})]


@pytest.fixture
def train_file(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in TRAIN))
    return str(path)


@pytest.mark.parametrize("name", ["oracle", "rules", "rules_plus", "logreg_checks", "logreg_memory", "knn"])
def test_every_classifier_returns_probabilities(name, train_file):
    panel = Panel([{"name": name, "train": train_file}])
    for r in TEST:
        probabilities = panel.classify(r)[name]
        assert list(probabilities) == ["none", "agent", "adversary"]
        assert np.isclose(sum(probabilities.values()), 1, atol=1e-4)


def test_oracle_and_rules_give_what_they_should(train_file):
    panel = Panel(["oracle", "rules"])
    for r in TEST:
        outputs = panel.classify(r)
        assert top_cause(outputs["oracle"]) == r["expected"]["cause"]
        assert top_cause(outputs["rules"]) == rule_cause(r["state"])


def test_trained_classifiers_need_training_rows_and_checkpoints_need_a_path():
    with pytest.raises(ValueError, match="train"):
        Panel(["logreg_checks"])
    with pytest.raises(ValueError, match="checkpoint"):
        Panel(["mlp"])
    with pytest.raises(ValueError, match="unknown classifier"):
        Panel(["nope"])
    assert set(CLASSIFIERS) == {"oracle", "rules", "rules_plus", "logreg_checks", "logreg_memory", "gbm_checks",
                                "gbm_memory", "knn", "knn_embed", "mlp", "mlp_embed", "stacking"}


def test_labels_let_one_classifier_appear_twice(train_file):
    panel = Panel([{"name": "logreg_checks", "train": train_file},
                   {"name": "logreg_checks", "label": "logreg_ngram", "train": train_file,
                    "injection_detector": "char_ngram"}])
    assert set(panel.classify(TEST[0])) == {"logreg_checks", "logreg_ngram"}
    with pytest.raises(ValueError, match="unique"):
        Panel(["rules", "rules"])


FLAG = {"none": 0.1, "agent": 0.8, "adversary": 0.1}
PASS = {"none": 0.9, "agent": 0.05, "adversary": 0.05}


@pytest.mark.parametrize("outputs, recover", [
    ({}, False),
    ({"a": FLAG}, True),
    ({"a": FLAG, "b": PASS}, False),          # 1 of 2 is not a majority
    ({"a": FLAG, "b": FLAG, "c": PASS}, True),
    ({"a": PASS, "b": PASS, "c": FLAG}, False),
])
def test_decide_is_a_strict_majority(outputs, recover):
    decision = decide(outputs)
    assert decision["recover"] is recover
    assert decision["flagged_by"] == [label for label, p in outputs.items() if p is FLAG]
