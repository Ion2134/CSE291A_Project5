import json

import numpy as np

from gbm import fit_gbm, fit_predict
from logreg import fit_logreg

OK = {"from_account_id": "a1", "payee_id": "p1", "amount_cents": 2500}
WRONG = {**OK, "amount_cents": 750}


def row(call, cause, group, received=None):
    state = (f"user_message: pay {group}\ntrusted: {{'from_account_id': 'a1', 'payee_id': 'p1', 'amount_cents': 2500}}\n"
             f"prior_calls:\ncall_index: 0\ntool: initiate_transfer\nmodel_call: {call}\n"
             f"received_args: {received or call}\noutput: {call}\n")
    return {"state": state, "expected": {"cause": cause, "output_correct": cause == "none"}, "questions": []}


def make_train():
    # min_samples_leaf is 10 by default, so give each class enough rows to form a leaf.
    return ([row(OK, "none", i) for i in range(20)] + [row(WRONG, "agent", i) for i in range(20, 40)]
            + [row(OK, "adversary", i, received={**OK, "payee_id": "p9"}) for i in range(40, 52)])


def test_learns_the_rule_and_returns_probabilities():
    model = fit_gbm(make_train())
    probabilities = model.predict([row(OK, "none", 90), row(WRONG, "agent", 91)])
    assert probabilities.shape == (2, 3) and np.allclose(probabilities.sum(axis=1), 1)
    assert probabilities.argmax(axis=1).tolist() == [0, 1]


def test_uses_exactly_the_logreg_features():
    train = make_train()
    assert fit_gbm(train).names == fit_logreg(train).names
    assert fit_gbm(train, memory=True).names == fit_logreg(train, memory=True).names
    assert fit_gbm(train, memory=True).names[-2:] == ["balance_unseen", "empty_transfer_output"]


def test_fit_is_deterministic_for_a_seed():
    train, test = make_train(), [row(OK, "none", 90), row(WRONG, "agent", 91)]
    assert np.array_equal(fit_gbm(train, seed=3).predict(test), fit_gbm(train, seed=3).predict(test))


def test_fit_predict_writes_the_tester_format(tmp_path):
    (tmp_path / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in make_train()))
    test = row(OK, "none", 90)
    test["state"] = test["state"].replace("pay 90", "send money to the landlord")  # its own request group
    (tmp_path / "test.jsonl").write_text(json.dumps(test) + "\n")
    result = fit_predict(tmp_path / "train.jsonl", tmp_path / "test.jsonl", tmp_path / "out", model="gbm_memory",
                         view="middleware")
    prediction = json.loads((tmp_path / "out/test_predictions.jsonl").read_text())
    assert set(prediction["cause_probabilities"]) == {"none", "agent", "adversary"}
    assert result["details"]["params"]["max_depth"] == 3
