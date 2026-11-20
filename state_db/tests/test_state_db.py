import json

import pytest

from state_db import StateStore


def ledger(balance, payments=()):
    return {"accounts": [{"account_id": "a1", "holder_name": "Alex", "balance_cents": balance, "is_primary": True}],
            "payees": [{"payee_id": "p1", "name": "Utilities", "routing_hint": "r1"}],
            "payments": [{"payment_id": pid, "from_account_id": "a1", "payee_id": "p1", "amount_cents": 100,
                          "status": "completed", "created_at": "t"} for pid in payments],
            "transactions": []}


@pytest.fixture
def store(tmp_path):
    store = StateStore(tmp_path / "state.db")
    yield store
    store.close()


def run_episode(store, episode_id):
    """start -> user message -> transfer of 100 -> tool reply. Returns the seq of each step."""
    start = store.start_episode(episode_id, ledger(1000))
    message = store.record_message(episode_id, 0, {"role": "user", "content": "pay p1 $1"})
    call = store.record_tool_call(episode_id, {"call_index": 0, "tool": "initiate_transfer",
                                               "model_call": {"amount_cents": 100}, "received_args": {"amount_cents": 100},
                                               "output": {"payment_id": "x1"}, "cause": "none", "output_correct": True},
                                  ledger(900, ["x1"]))
    reply = store.record_message(episode_id, 1, {"role": "tool", "content": "done"})
    return start, message, call, reply


def test_reads_return_the_state_as_of_a_step(store):
    start, message, call, reply = run_episode(store, "ep1")
    assert store.ledger_at("ep1", message)["accounts"][0]["balance_cents"] == 1000
    after = store.ledger_at("ep1", call)
    assert after["accounts"][0]["balance_cents"] == 900 and after["payments"][0]["payment_id"] == "x1"
    assert [m["role"] for m in store.messages_at("ep1", reply)] == ["user", "tool"]
    assert store.tool_calls_at("ep1", call)[0]["received_args"] == {"amount_cents": 100}
    assert store.messages_at("ep1", start) == []


def test_delete_after_removes_only_that_episodes_later_steps(store):
    _, message, _, _ = run_episode(store, "ep1")
    _, _, other_call, other_reply = run_episode(store, "ep2")
    assert store.delete_after("ep1", message) == 2
    assert store.ledger_at("ep1", 10**9)["accounts"][0]["balance_cents"] == 1000
    assert store.tool_calls_at("ep1", 10**9) == []
    assert [m["role"] for m in store.messages_at("ep1", 10**9)] == ["user"]
    # ep2 is untouched.
    assert store.ledger_at("ep2", other_reply)["accounts"][0]["balance_cents"] == 900
    # New steps after a deletion carry on with fresh seqs.
    assert store.record_message("ep1", 1, {"role": "tool", "content": "blocked"}) > other_reply


def test_seq_at_maps_wall_time_to_the_last_step(store):
    start, message, call, reply = run_episode(store, "ep1")
    times = dict(store._db.execute("SELECT seq, wall_time FROM steps"))
    assert store.seq_at("ep1", times[call]) == call
    assert store.seq_at("ep1", times[start] - 1) is None
    assert store.seq_at("ep1", times[reply] + 60) == reply
