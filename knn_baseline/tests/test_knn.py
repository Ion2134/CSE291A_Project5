import numpy as np

from knn import knn_probabilities

OK = {"from_account_id": "a1", "payee_id": "p1", "amount_cents": 2500}
WRONG = {**OK, "amount_cents": 750}


def row(call, cause, group):
    state = (f"user_message: pay {group}\ntrusted: {{'from_account_id': 'a1', 'payee_id': 'p1', 'amount_cents': 2500}}\n"
             f"prior_calls:\ncall_index: 0\ntool: initiate_transfer\nmodel_call: {call}\n"
             f"received_args: {call}\noutput: {call}\n")
    return {"state": state, "expected": {"cause": cause, "output_correct": cause == "none"}, "questions": []}


def test_probabilities_are_valid_and_follow_the_neighbours():
    train = [row(OK, "none", i) for i in range(6)] + [row(WRONG, "agent", i) for i in range(6, 12)]
    probabilities = knn_probabilities(train, [row(OK, "none", 20), row(WRONG, "agent", 21)], neighbors=3,
                                      use_checks=True)
    assert probabilities.shape == (2, 3) and np.allclose(probabilities.sum(axis=1), 1)
    assert probabilities.argmax(axis=1).tolist() == [0, 1]


def test_more_neighbours_than_rows_is_fine():
    train = [row(OK, "none", 0), row(WRONG, "agent", 1)]
    assert knn_probabilities(train, [row(OK, "none", 2)], neighbors=50).shape == (1, 3)
