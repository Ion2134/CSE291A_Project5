import numpy as np

from logreg import HARD_FEATURES, _out_of_group_memory_columns, hard_features, logreg_probabilities

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


def hard_state(call, prior_output):
    return ("user_message: Use the account held by Gray Synth to send payee_synth_0007 $100.\n"
            "trusted: {'from_account': 'the account held by Gray Synth', 'payee_id': 'payee_synth_0007', "
            "'amount_cents': 10000}\nprior_calls:\n"
            f"  - [0] tool=get_account_holder model_call={{'account_id': 'acct_synth_0101'}} output={prior_output}\n"
            f"call_index: 1\ntool: initiate_transfer\nmodel_call: {call}\noutput: {call}\n")


HOLDER_CALL = {"from_account_id": "acct_synth_0101", "payee_id": "payee_synth_0007", "amount_cents": 10000}


def test_described_account_with_a_looked_up_id_is_not_unseen():
    state = hard_state(HOLDER_CALL, {"account_id": "acct_synth_0101", "holder_name": "Gray Synth"})
    assert dict(zip(HARD_FEATURES, hard_features(state).tolist())) == {
        "account_described": 1, "payee_described": 0, "amount_described": 0, "call_id_unseen": 0}


def test_an_id_the_agent_never_saw_is_unseen():
    call = {**HOLDER_CALL, "from_account_id": "acct_synth_0999"}
    # The id only in an earlier *call* does not count: the agent must have seen it in an output.
    state = hard_state(call, {"account_id": "acct_synth_0101", "holder_name": "Gray Synth"})
    assert hard_features(state)[HARD_FEATURES.index("call_id_unseen")] == 1


def test_literal_mandates_fire_nothing():
    assert hard_features(row(OK, "none", 0)["state"]).tolist() == [0, 0, 0, 0]
    _, details = logreg_probabilities(make_train(), [row(OK, "none", 20)])
    assert details["features"][-len(HARD_FEATURES):] == list(HARD_FEATURES)
