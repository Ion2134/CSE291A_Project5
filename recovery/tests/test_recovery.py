import pytest

from recovery import BLOCKED_REPLY, RecoveryPolicy, rollback

FLAGGED = {"recover": True, "flagged_by": ["a", "b"]}
NOT_FLAGGED = {"recover": False, "flagged_by": ["a"]}


@pytest.mark.parametrize("stage, classifiers, recover, reason", [
    ("request", FLAGGED, False, "blocked_before_tool"),   # the tool never ran: nothing to undo
    ("response", None, True, "response_sig_failed"),      # the tool ran, its output was tampered with
    ("response", NOT_FLAGGED, True, "response_sig_failed"),
    (None, FLAGGED, True, "classifiers"),
    (None, NOT_FLAGGED, False, "not_flagged"),
    (None, None, False, "no_classifiers"),
])
def test_decision_rules(stage, classifiers, recover, reason):
    decision = RecoveryPolicy().decide(stage, classifiers)
    assert (decision.recover, decision.reason) == (recover, reason)


def test_cap_stops_rollbacks_but_still_reports_why():
    policy = RecoveryPolicy(max_recoveries=1)
    assert policy.decide(None, FLAGGED).recover
    capped = policy.decide(None, FLAGGED)
    assert (capped.recover, capped.reason, capped.flagged_by) == (False, "cap_reached", ["a", "b"])
    # Calls that would not be rolled back anyway keep their own reason.
    assert policy.decide("request", None).reason == "blocked_before_tool"
    assert RecoveryPolicy(max_recoveries=0).decide("response", None).reason == "cap_reached"


class FakeStore:
    def __init__(self):
        self.calls = []

    def delete_after(self, episode_id, seq):
        self.calls.append(("delete_after", episode_id, seq))
        return 3

    def ledger_at(self, episode_id, seq):
        return {"accounts": [{"account_id": "a1", "balance_cents": 1000}]}

    def messages_at(self, episode_id, seq):
        return [{"role": "user", "content": "pay"}]


def test_rollback_deletes_then_reads_the_state_at_that_step():
    store = FakeStore()
    result = rollback(store, "ep1", 7)
    assert store.calls == [("delete_after", "ep1", 7)]
    assert result.deleted_steps == 3 and result.ledger["accounts"][0]["balance_cents"] == 1000
    assert result.messages == [{"role": "user", "content": "pay"}]
    assert "error" in BLOCKED_REPLY


def test_rollback_against_a_real_state_store(tmp_path):
    state_db = pytest.importorskip("state_db")
    store = state_db.StateStore(tmp_path / "state.db")
    ledger = lambda balance: {"accounts": [{"account_id": "a1", "holder_name": "A", "balance_cents": balance,
                                           "is_primary": True}], "payees": [], "payments": [], "transactions": []}
    store.start_episode("ep", ledger(1000))
    before = store.record_message("ep", 0, {"role": "assistant", "content": "calling transfer"})
    store.record_tool_call("ep", {"call_index": 0, "tool": "initiate_transfer"}, ledger(400))
    result = rollback(store, "ep", before)
    assert result.ledger["accounts"][0]["balance_cents"] == 1000 and result.deleted_steps == 1
    assert store.tool_calls_at("ep", 10**9) == []
    store.close()
