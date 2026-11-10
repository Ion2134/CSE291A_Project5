import numpy as np

from logreg import _out_of_group_memory_columns, logreg_probabilities

OK = {"from_account_id": "a1", "payee_id": "p1", "amount_cents": 2500}
WRONG = {**OK, "amount_cents": 750}


def row(call, cause, group, received=None):
    state = (f"user_message: pay {group}\ntrusted: {{'from_account_id': 'a1', 'payee_id': 'p1', 'amount_cents': 2500}}\n"
             f"prior_calls:\ncall_index: 0\ntool: initiate_transfer\nmodel_call: {call}\n"
             f"received_args: {received or call}\noutput: {call}\n")
    return {"state": state, "expected": {"cause": cause, "output_correct": cause == "none"}, "questions": []}


def make_train():
    return ([row(OK, "none", i) for i in range(6)] + [row(WRONG, "agent", i) for i in range(6, 12)]
            + [row(OK, "adversary", 12, received={**OK, "payee_id": "p9"})])


def test_learns_the_rule_and_returns_probabilities():
    probabilities, details = logreg_probabilities(make_train(), [row(OK, "none", 20), row(WRONG, "agent", 21)])
    assert np.allclose(probabilities.sum(axis=1), 1)
    assert probabilities.argmax(axis=1).tolist() == [0, 1]
    assert "call_any_mismatch" in details["features"]


def test_memory_columns_exist_and_leave_out_own_group():
    train = make_train()
    assert _out_of_group_memory_columns(train).shape == (len(train), 2)
    probabilities, details = logreg_probabilities(train, [row(OK, "none", 20)], memory=True)
    assert details["features"][-2:] == ["balance_unseen", "empty_transfer_output"]
    assert probabilities.shape == (1, 3)
