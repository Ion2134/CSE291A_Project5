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


def test_a_verified_mandate_outranks_the_classifiers():
    # Even a unanimous classifier vote can't undo a call the user's signed mandate allowed.
    decision = RecoveryPolicy().decide(None, FLAGGED, mandate_verified=True)
    assert (decision.recover, decision.reason) == (False, "mandate_verified")
    # The other rules are unchanged: a refused call has nothing to undo, a tampered response is rolled back.
    assert RecoveryPolicy().decide("request", None, mandate_verified=False).reason == "blocked_before_tool"
    assert RecoveryPolicy().decide("response", None, mandate_verified=True).reason == "response_sig_failed"
    policy = RecoveryPolicy(max_recoveries=1)
    policy.decide(None, FLAGGED, mandate_verified=True)
    assert policy.recoveries == 0  # a skipped vote doesn't use up the cap


def test_single_use_mandate_keeps_a_verified_call_whose_reply_failed():
    # The call is what the user signed for and can't be repeated under that mandate: keep it.
    kept = RecoveryPolicy().decide("response", None, mandate_verified=True, single_use=True)
    assert (kept.recover, kept.reason) == (False, "verified_reply_withheld")
    # Without single use a retry could pay twice, so the call is rolled back as before.
    assert RecoveryPolicy().decide("response", None, mandate_verified=True).reason == "response_sig_failed"
    assert RecoveryPolicy().decide("response", None, single_use=True).reason == "response_sig_failed"


def test_the_agents_own_refused_call_may_be_retried_only_when_allowed():
    retry = RecoveryPolicy().decide("request", None, own_call_refused=True, retry_refused=True)
    assert (retry.recover, retry.reason) == (False, "retry_allowed")
    # Retries off, or a refusal that isn't the agent's own signed call (tampering, replay): no retry.
    assert RecoveryPolicy().decide("request", None, own_call_refused=True).reason == "blocked_before_tool"
    assert RecoveryPolicy().decide("request", None, retry_refused=True).reason == "blocked_before_tool"
    # Retries share the cap, so an agent that keeps getting it wrong stops.
    policy = RecoveryPolicy(max_recoveries=2)
    reasons = [policy.decide("request", None, own_call_refused=True, retry_refused=True).reason for _ in range(3)]
    assert reasons == ["retry_allowed", "retry_allowed", "cap_reached"]


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
